import torch
from torch import nn

import numpy as np
import ot

from dingo.core.utils import torchutils
from .cflow_base import ContinuousFlowsBase


class FlowMatching(ContinuousFlowsBase):
    """
    Class for continuous normalizing flows trained with flow matching.

        t               ~ U[0, 1-eps)                               noise level
        theta_0         ~ N(0, 1)                                   sampled noise
        theta_1         = theta                                     pure sample
        theta_t         = c1(t) * theta_1 + c0(t) * theta_0         noisy sample

        eps             = 0
        c0              = (1 - (1 - sigma_min) * t)
        c1              = t

        v_target        = theta_1 - (1 - sigma_min) * theta_0
        loss            = || v_target - network(theta_t, t) ||
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eps = 0
        self.sigma_min = self.model_kwargs["posterior_kwargs"]["sigma_min"]
        self.match_rate = self.model_kwargs["posterior_kwargs"].get("match_rate", None)
        self.match_type = self.model_kwargs["posterior_kwargs"].get("match_type", None)
        self.sinkhorn_reg = self.model_kwargs["posterior_kwargs"].get("sinkhorn_reg", None)
        self.beta = 1.0
        if self.match_rate is not None and self.match_type not in ["pcot", "scot"]:
            raise ValueError("Match rate can only be defined for pcot and scot.")


    def evaluate_vectorfield(self, t, theta_t, *context_data):
        """
        Vectorfield that generates the flow, see Docstring in ContinuousFlowsBase for
        details. With flow matching, this vectorfield is learnt directly.
        """
        # If t is a number (and thus the same for each element in this batch),
        # expand it as a tensor. This is required for the odeint solver.
        t = t * torch.ones(len(theta_t), device=theta_t.device)
        return self.network(t, theta_t, *context_data)

    def loss(self, theta, *context_data):
        """
        Calculates loss as the the mean squared error between the predicted vectorfield
        and the vector field for transporting the parameter data to samples from the
        prior.

        Parameters
        ----------
        theta: torch.tensor
            parameters (e.g., binary-black hole parameters)
        *context_data: list[torch.Tensor]
            context data (e.g., gravitational-wave data)

        Returns
        -------
        torch.tensor
            loss tensor
        """
        # Shall we allow for multiple time evaluations for every data, context pair (to improve efficiency)?
        mse = nn.MSELoss()

        t = self.sample_t(len(theta))
        theta_0 = self.sample_theta_0(len(theta))
        theta_1 = theta

        # reorder params if necessary
        n_match = None
        if self.match_type == "pcot":
            theta_0, n_match = self.reorder_batch(
                params=theta_1,
                init_params=theta_0,
                obs=theta_0,
                beta=self.beta,
                param_condition=True,
                sinkhorn_reg=self.sinkhorn_reg,
            )

        # eval the embedding
        context_embedding = torchutils.forward_pass_with_unpacked_tuple(
            self.network.context_embedding_net, *context_data
        )

        # reorder if necessary
        if self.match_type == "scot":
            theta_0, n_match = self.reorder_batch(
                params=theta,
                init_params=theta_0,
                obs=context_embedding,
                beta=self.beta,
                param_condition=False,
                sinkhorn_reg=self.sinkhorn_reg,
            )

        # interpolate
        theta_t = ot_conditional_flow(theta_0, theta_1, t, self.sigma_min)
        true_vf = theta - (1 - self.sigma_min) * theta_0

        # get the theta and t embedding
        # embed theta (self.embedding_net_theta might just be identity)
        t_and_theta_embedding = torch.cat((t.unsqueeze(1), theta_t), dim=1)
        t_and_theta_embedding = self.network.theta_embedding_net(t_and_theta_embedding)

        predicted_vf = self.network.forward_from_embedding(context_embedding, t_and_theta_embedding)
        loss = mse(predicted_vf, true_vf)

        # adapt beta if necessary
        if self.match_rate is not None and self.network.training:
            current_match_rate = n_match / len(theta_0)
            if current_match_rate < self.match_rate:
                self.beta = self.beta * 1.1
            else:
                self.beta = self.beta * 0.9

        return loss

    @staticmethod
    @torch.no_grad()
    def reorder_batch(
            params: torch.Tensor,
            init_params: torch.Tensor,
            obs: torch.Tensor,
            beta: None | float = None,
            param_condition: bool = False,
            sinkhorn_reg: None | float = None,
    ) -> tuple[torch.Tensor, int]:
        """
        Reorders a batch of initial params according to the optimal transport plan
        :param params: The target params
        :param init_params: The initial params
        :param obs: The corresponding observations (or the embedding of it)
        :param beta: The beta value for the obs anchoring, if None, no reordering is done.
        :param param_condition: If true the parameters are used for anchoring, if false the observations.
        :param sinkhorn_reg: The regularization parameter for the sinkhorn algorithm, if None the exact OT is used.
        :return: The reordered batch of initial params
        """

        # nothing to do
        if beta is None:
            return init_params, init_params.shape[0]

        # get the shapes of the params and the obs
        _, param_dim = params.shape
        _, obs_dim = obs.shape

        # dist matrices
        dist_matrix_params = torch.cdist(params, init_params, p=2).square() / param_dim
        # choose the right distance matrix
        if param_condition:
            dist_matrix_obs = torch.cdist(params, params, p=2).square() / param_dim
        else:
            dist_matrix_obs = torch.cdist(obs, obs, p=2).square() / obs_dim

        # add according to beta
        dist_matrix = dist_matrix_params + beta * dist_matrix_obs

        if sinkhorn_reg is None:
            # get the optimal transport plan
            g0 = ot.emd([], [], dist_matrix.cpu().numpy())
            tp = np.argwhere(g0)

            # reorder the initial params
            init_params_reorder = init_params[tp[:, 1]]
            return init_params_reorder, np.sum(tp[:, 1] == tp[:, 0])

        else:
            # sinkhorn on GPU
            p = ot.sinkhorn([], [], dist_matrix, reg=sinkhorn_reg)
            p /= p.sum(dim=1, keepdim=True)

            # sample on element per row
            cum_p = torch.cumsum(p, axis=1)
            r = torch.rand(len(cum_p), 1, device=p.device)
            selection = r < cum_p
            indices = torch.argmax(selection.float(), dim=1)

            # reorder the initial params
            init_params_reorder = init_params[indices]

            return init_params_reorder, torch.sum(indices == torch.arange(len(indices), device=p.device)).item()


def ot_conditional_flow(x_0, x_1, t, sigma_min):
    return (1 - (1 - sigma_min) * t)[:, None] * x_0 + t[:, None] * x_1
