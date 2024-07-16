import torch
from torch import nn

import dingo.core.utils as utils
from dingo.core.posterior_models.base_model import Base
from dingo.core.nn.enets import DenseResidualNet
from dingo.core.utils import torchutils


class InfoMax(Base):
    """
    The info max model as base class
    """

    def __init__(self, context_embedding_net, **kwargs):
        # the context embedding network
        self.context_embedding_net = context_embedding_net

        # get the summary network kwargs (set in init)
        self.summary_network_kwargs = None

        super().__init__(**kwargs)

    def initialize_network(self):
        train_settings = self.metadata["train_settings"]
        theta_dim = train_settings["model"]["posterior_kwargs"]["input_dim"]
        pretrain_settings = train_settings["training"]["pretrain"]
        self.summary_network_kwargs = pretrain_settings["summary_net"]
        self.summary_network_kwargs["theta_dim"] = theta_dim

        self.network = InfoMaxNetwork(context_embedding_net=self.context_embedding_net,
                                      summary_network_kwargs=self.summary_network_kwargs)

    def sample_batch(self, *context_data):
        raise NotImplementedError

    def sample_and_log_prob_batch(self, *context_data):
        raise NotImplementedError

    def log_prob_batch(self, data, *context_data):
        raise NotImplementedError

    def loss(self, theta: torch.Tensor, obs: torch.Tensor, m: int = 10) -> torch.Tensor:
        """
        Computes the mutual information loss according to arXiv:2010.10079

        Parameters
        ----------
        theta: torch.Tensor
            The parameters (theta) of the distribution
        obs: torch.Tensor
            The observation that will be embedded
        m: int
            The number of cycles for the independent part

        Returns
        -------
        An estimate of the mutual information loss
        """

        # the joint distribution
        obs_emb = self.context_embedding_net(obs)
        x_in = torch.cat([theta, obs_emb], dim=1)
        joint = self.network.summary_net(x_in)
        loss_joint = -torch.mean(torch.log(1 + torch.exp(-joint)))

        # independent part
        batch_size = theta.shape[0]
        param_index = torch.randint(0, batch_size, (m * batch_size,))
        obs_index = torch.randint(0, batch_size, (m * batch_size,))
        x_in = torch.cat([theta[param_index], obs_emb[obs_index]], dim=1)
        independent = self.network.summary_net(x_in)
        loss_independent = torch.mean(torch.log(1 + torch.exp(independent)))

        # return the loss
        loss = loss_joint - loss_independent

        return -loss

class InfoMaxNetwork(nn.Module):
    """
    This class implements the InfoMax model which pretrains the embedding network with an information maximizing loss.
    """

    def __init__(self, context_embedding_net: nn.Module, summary_network_kwargs: dict):
        """
        Inits the InfoMax model
        :param context_embedding_net: The context embedding network
        :param summary_network_kwargs: The dict containing the kwargs for the summary network
        """

        super().__init__()
        self.context_embedding_net = context_embedding_net
        self.summary_network_kwargs = summary_network_kwargs

        # get the dimension of the summary dimension
        self.summary_dim = context_embedding_net[-1].output_dim
        activation_fn = torchutils.get_activation_function_from_string(
            summary_network_kwargs["activation"]
        )
        self.summary_net = DenseResidualNet(
            input_dim=self.summary_dim + summary_network_kwargs["theta_dim"],
            output_dim=summary_network_kwargs["output_dim"],
            hidden_dims=summary_network_kwargs["hidden_dims"],
            activation=activation_fn,
            dropout=0.0,
            batch_norm=summary_network_kwargs.get("batch_norm", False),
        )

    def forward(self, obs: torch.Tensor, theta: torch.Tensor):
        """
        Performs a forward pass for the joint probability of summaries and parameters

        Parameters
        ----------
        obs: torch.Tensor
            The summaries of the observations
        theta: torch.Tensor
            The parameters

        Returns
        -------
        The output of the model
        """

        summaries = self.context_embedding_net(obs)
        obs = torch.cat([summaries, theta], dim=-1)
        return self.summary_net(obs)