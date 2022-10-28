import torch
from torch import nn
import numpy as np


class TPS(nn.Module):
    """
    TPS encoder and decoder
    """
    
    def __init__(self, num_fiducial=8, fiducial_shape=(0.25, 1), grid_size=(32, 100), num_points=40,
                 fiducial_type="edge"):
        """Generate L and inv_delta_C for later."""
        super().__init__()
        self.eps = 1e-6
        self.fiducial_height = fiducial_shape[0]
        self.fiducial_width = fiducial_shape[1]
        self.num_fiducial = num_fiducial
        assert fiducial_type in ["edge", "cross", "center"]
        self.fiducial_type = fiducial_type
        if self.fiducial_type == "edge":
            C = self._build_C_edge(num_fiducial)  # num_fiducial x 2
        elif self.fiducial_type == "cross":
            C = self._build_C_cross()
        else:
            C = self._build_C_center()
        self.C = C  # num_fiducial个基准点，即P
        # self.C = np.stack([C, C[:,[1,0]]])
        self.num_points = num_points
        self.P = self._build_P(num_points)
        # self.P = np.stack([P, P[:,[1,0]]])
        # for multi-gpu, you need register buffer
        inv_delta_C = torch.tensor(self._build_inv_delta_C(self.num_fiducial, C)).float()
        self.register_buffer('inv_delta_C', inv_delta_C)
        
        L = torch.tensor(self._build_L(self.num_fiducial, self.C, self.P)).float()  # 基准矩形上采样20个点和8个基准点之间计算L:n x (num_fiducial+3)
        # self.L = L # n x num_fiducial+3
        self.register_buffer("L", L)
        self.grid_size = grid_size
        P_grid = self._build_P_grid(*grid_size)
        L_grid = torch.tensor(self._build_L(self.num_fiducial, self.C, P_grid)).float()
        # self.L_grid = L_grid
        self.register_buffer("L_grid", L_grid)
    
    def _build_C_edge(self, num_fiducial):
        n = num_fiducial // 2
        ctrl_pts_x = np.linspace(-1.0, 1.0, n) * self.fiducial_width
        ctrl_pts_y_top = -1 * np.ones(n) * self.fiducial_height
        ctrl_pts_y_bottom = np.ones(n) * self.fiducial_height
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        # if not self.head_tail:
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # num_fiducial x 2
    
    def _build_C_cross(self):
        
        ctrl_pts_x = np.linspace(-1.0, 1.0, 3) * self.fiducial_width
        ctrl_pts_y_top = -1 * np.ones(3) * self.fiducial_height
        ctrl_pts_y_bottom = np.ones(3) * self.fiducial_height
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_pts_x_center = np.linspace(-1.0, 1.0, 5)[[1, 3]] * self.fiducial_width
        ctrl_pts_y_center = np.zeros(2)
        ctrl_pts_center = np.stack([ctrl_pts_x_center, ctrl_pts_y_center], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_center, ctrl_pts_bottom], axis=0)
        return C
    
    def _build_C_center(self):
        n = 6
        ctrl_pts_x = np.linspace(-1.0, 1.0, n) * self.fiducial_width
        ctrl_pts_y_top = -1 * np.ones(n) * self.fiducial_height
        ctrl_pts_y_bottom = np.ones(n) * self.fiducial_height
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        center_line = (ctrl_pts_top + ctrl_pts_bottom) / 2
        center_line = center_line[1:-1]
        C = np.concatenate([ctrl_pts_top[[0, -1]], center_line, ctrl_pts_bottom[[0, -1]]])
        return C  # num_fiducial x 2
    
    def _build_P_grid(self, h, w):
        fiducial_grid_x = np.linspace(-1, 1, w) * self.fiducial_width
        fiducial_grid_y = np.linspace(-1, 1, h) * self.fiducial_height
        P = np.stack(np.meshgrid(fiducial_grid_x, fiducial_grid_y), axis=2)  # self.fiducial_w x self.fiducial_h x 2
        return P.reshape([-1, 2])
    
    def _build_inv_delta_C(self, num_fiducial, C):
        """Return inv_delta_C which is needed to calculate T."""
        hat_C = np.zeros((num_fiducial, num_fiducial), dtype=float)
        for i in range(0, num_fiducial):
            for j in range(i, num_fiducial):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        delta_C = np.concatenate(  # num_fiducial+3 x num_fiducial+3
            [
                np.concatenate([np.ones((num_fiducial, 1)), C, hat_C], axis=1),  # num_fiducial x num_fiducial+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x num_fiducial+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, num_fiducial))], axis=1)  # 1 x num_fiducial+3
            ],
            axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # num_fiducial+3 x num_fiducial+3
    
    def _build_P(self, num_pts):
        fiducial_grid_x = np.linspace(-1.0, 1.0, int(num_pts / 2)) * self.fiducial_width
        ctrl_pts_y_top = -1 * np.ones(fiducial_grid_x.shape[0]) * self.fiducial_height
        ctrl_pts_y_bottom = np.ones(fiducial_grid_x.shape[0]) * self.fiducial_height
        ctrl_pts_top = np.stack([fiducial_grid_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([fiducial_grid_x, ctrl_pts_y_bottom], axis=1)
        P = np.concatenate([ctrl_pts_top, ctrl_pts_bottom[::-1]], axis=0)
        return P.reshape([-1, 2])  # n (= self.fiducial_width x self.fiducial_height) x 2
    
    def _build_L(self, num_fiducial, C, P):
        n = P.shape[0]  # n (= self.fiducial_width x self.fiducial_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, num_fiducial, 1))  # n x 2 -> n x 1 x 2 -> n x num_fiducial x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x num_fiducial x 2
        P_diff = P_tile - C_tile  # n x num_fiducial x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x num_fiducial
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x num_fiducial
        L = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return L  # n x num_fiducial+3
    
    def build_inv_P(self, num_fiducial, P, C):
        L = self._build_L(num_fiducial, C, P)  # n x (num_fiducial +3)
        L = np.concatenate([L, np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x num_fiducial+3
                            np.concatenate([np.zeros((1, 3)), np.ones((1, num_fiducial))], axis=1)  # 1 x num_fiducial+3
                            ])
        inv_L = np.linalg.pinv(L)  # (num_fiducial +3) x (n +3)
        return inv_L
    
    def solve_T(self, batch_C_prime, batch_P=None):
        device = self.inv_delta_C.device
        if batch_P is None:  # solve with control point pair
            batch_size = batch_C_prime.shape[0]
            batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        
        else:  # solve with least square method
            batch_size = batch_C_prime.size(0)
            batch_inv_delta_C = torch.from_numpy(self.build_inv_P(self.num_fiducial, batch_P, self.C))[None].repeat(batch_size, 1, 1).float()
        if not isinstance(batch_C_prime, torch.Tensor):
            batch_C_prime = torch.from_numpy(batch_C_prime)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)), dim=1)  # batch_size x num_fiducial+3 x 2
        batch_Q = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x num_fiducial+3 x 2
        return batch_Q
    
    def build_P_border(self, batch_Q):
        batch_Q = batch_Q.view(-1, self.num_fiducial + 3, 2)
        batch_L = self.L.repeat(batch_Q.shape[0], 1, 1)  # L
        batch_Y = torch.bmm(batch_L, batch_Q)  # LQ=Y
        return batch_Y
    
    def build_P_grid(self, batch_Q):
        batch_Q = batch_Q.view(-1, self.num_fiducial + 3, 2)
        batch_L_grid = self.L_grid.repeat(batch_Q.shape[0], 1, 1)
        batch_P_grid = torch.bmm(batch_L_grid, batch_Q)
        return batch_P_grid  # batch_size x n x 2
