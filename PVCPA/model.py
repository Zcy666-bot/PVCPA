import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class LoadExcelDataset(Dataset):
    def __init__(self, start_date, duration, Area_Gantt_chart, sorts, point, area):

        self.A = Area_Gantt_chart
        self.start_date = start_date
        self.duration = duration
        self.sorts = sorts
        self.point = point
        self.area = area

        input_deformation = pd.DataFrame()
        temp = pd.read_excel('./Deformation.xlsx','Sheet1(15)',  usecols='B:AT')
        input_deformation = input_deformation.append(temp)
        del temp
        self.Input_Deformation = np.array(input_deformation, dtype="float32")

        maximum_array_X = np.empty((int(self.duration * 45), 3), dtype="float32")
        maximum_array_Y = np.empty((int(self.duration * 45), 1), dtype="float32")
        maximum_array_Date = np.empty((int(self.duration * 45), 1), dtype="int")
        current_row = 0

        if self.sorts == 'train':
            for d_i in range(self.duration):
                for a_i in range(6):
                    if self.A[d_i, a_i] == 1:
                        if a_i < 3:
                            z_position_i = 0
                        else:
                            z_position_i = 1

                        if a_i % 3 == 0:
                            y_position_i = 0
                            A_i = 0
                        elif a_i % 3 == 1:
                            y_position_i = 1
                            A_i = 15
                        else:
                            y_position_i = 2
                            A_i = 30

                        for b_l in range(15):
                            maximum_array_X[current_row, :] = [b_l*6+8, y_position_i, z_position_i]
                            maximum_array_Y[current_row, 0] = self.Input_Deformation[d_i, int(A_i+b_l)]
                            maximum_array_Date[current_row, 0] = d_i
                            current_row += 1

        elif self.sorts == 'test':
            y_position_i = self.area
            A_l_i = self.area * 15
            for d_l in range(self.duration):
                date_i = int(d_l + self.start_date)
                if self.A[date_i, self.area] == 1:
                    z_position_i = 0
                elif self.A[date_i, self.area + 3] == 1:
                    z_position_i = 1
                else:
                    z_position_i = 1

                maximum_array_X[current_row, :] = [self.point * 6 + 2, y_position_i, z_position_i]
                maximum_array_Y[current_row, 0] = self.Input_Deformation[date_i, int(A_l_i + self.point-1)]
                maximum_array_Date[current_row, 0] = date_i
                current_row += 1

        self.Input_X = maximum_array_X[:current_row]
        self.Output_Y = maximum_array_Y[:current_row]
        self.Input_Date = maximum_array_Date[:current_row]

    def __len__(self):
        return self.Input_X.shape[0]

    def __getitem__(self, idx):
        date_gotten = self.Input_Date[idx]
        x_gotten = self.Input_X[idx, :]
        y_gotten = self.Output_Y[idx]
        return x_gotten, date_gotten, y_gotten

def readmyexcel(path, sheet_name, cols):
    dataSet = pd.DataFrame()
    data = pd.read_excel(path, sheet_name=sheet_name, usecols=cols)
    dataSet = dataSet.append(data)
    del data
    data_nparray = np.array(dataSet, dtype="float32")
    headers = dataSet.columns.tolist()
    return data_nparray, headers

def BC_point_multiplication(defor_15, cond_12):
    defor_BC = np.empty((20, 3, 6), dtype="float32")
    x_BC = np.empty((20, 3, 6), dtype="float32")

    for d_pm in range(20):
        for a_pm in range(3):
            defor_17_da = np.zeros(17)
            defor_17_da[1: 16] = defor_15[d_pm, (a_pm*15):(a_pm*15+15)]
            x_17_da = np.array([0, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92, 100])

            interp_func = Akima1DInterpolator(x_17_da, defor_17_da)
            x_1001_da = np.arange(0, 100.1, 0.1)
            defor_1001_da = interp_func(x_1001_da)

            excavation_da = 60
            struct_da = 40
            if Area_Gantt_chart[d_pm, a_pm]==1:
                excavation_da = cond_12[d_pm, a_pm]
                struct_da = cond_12[d_pm, a_pm + 6]
            gap_da = excavation_da - struct_da

            for i_pm in range(6):
                x_i_pm = excavation_da - i_pm * (gap_da//0.5) * 0.1
                x_BC[d_pm, a_pm, i_pm] = x_i_pm
                defor_BC[d_pm, a_pm, i_pm] = defor_1001_da[int(x_i_pm*10)]

    x_BC = torch.from_numpy(x_BC).to(device)
    defor_BC = torch.from_numpy(defor_BC).to(device)
    return x_BC, defor_BC

class LSTM_Att(nn.Module):
    def __init__(self, input_dim=164, hidden_size=64):
        super(LSTM_Att, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,hidden_size=hidden_size, num_layers=2,
            bidirectional=True, batch_first=True, dropout=0.5)
        self.tanh = nn.Tanh()
        self.m = nn.Parameter(torch.zeros(hidden_size * 2))
        nn.init.uniform_(self.m, a=-0.5, b=0.5)
        self.FC_for_timefeature = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.Linear(hidden_size * 4, 20),
        )

    def forward(self, x):
        t, _ = self.lstm(x)
        a = self.tanh(t)
        alpha = F.softmax(torch.matmul(a, self.m), dim=1).unsqueeze(-1)
        f = t * alpha
        f = torch.sum(f, 1)
        y = self.FC_for_timefeature(f)
        return y

class Parameter_Modification(nn.Module):
    def __init__(self, hidden_size=64):
        super(Parameter_Modification, self).__init__()
        self.out_feature = 6
        self.fc_AB = nn.Sequential(
            nn.ReLU(),
            nn.Linear(26, hidden_size*2),
            nn.Linear(hidden_size*2, self.out_feature),
        )
        self.fc_CD = nn.Sequential(
            nn.ReLU(),
            nn.Linear(26, hidden_size*2),
            nn.Linear(hidden_size*2, self.out_feature),
        )

    def forward(self, paras, conds, segm):
        x = torch.cat((conds, paras), dim=1)
        if segm == 'AB':
            modify_paras = self.fc_AB(x)
        elif segm == 'BC':
            modify_paras = x[:, (-self.out_feature):]
        elif segm == 'CD':
            modify_paras = self.fc_CD(x)
        elif segm == 'undistinguished':
            modify_paras = torch.zeros(x.shape[0], self.out_feature).to(device)
            for i in range(x.shape[0]):
                x_i = x.clone()
                x_i = x_i[i].unsqueeze(0)
                if x_i[0, 3] < x_i[0, 2]:
                    modify_paras[i] = self.fc_AB(x_i)
                elif (x_i[0, 3] >= x_i[0, 2]) and (x_i[0, 3] <= x_i[0, 1]):
                    modify_paras[i] = x_i[:, (-self.out_feature):]
                elif (x_i[0, 3] > x_i[0, 1]):
                    modify_paras[i] = self.fc_CD(x_i)
        return modify_paras

def choose_data(basic_data, date_c, cate):
    num_c = date_c.shape[0]
    nparray = basic_data.detach().cpu().numpy()
    datas = np.empty((num_c, timesteps, int(3+Items_Conds.shape[1])),
                     dtype="float32")

    for b_c in range(num_c):
        for t_c in range(timesteps):
            if cate=='test':
                date_t_c = history_dates-timesteps+t_c
            else:
                date_t_c = int(date_c[b_c] - timesteps + t_c)

            datas[b_c, t_c, 0] = nparray[b_c, 0] / 100
            datas[b_c, t_c, 1] = nparray[b_c, 1] / 2
            datas[b_c, t_c, 2] = nparray[b_c, 2] / 1

            if date_t_c >= 0:
                datas[b_c, t_c, 3:] = Items_Conds[date_t_c, :]
            else:
                datas[b_c, t_c, 3:] = Items_Conds[0, :]

    choose_data = torch.from_numpy(datas).float().to(device)
    return choose_data

def processing_point_date_information(x_point_p, date_p, point_or_boundary):
    num_p = date_p.shape[0]
    infor_p = np.empty((num_p, 6), dtype="float32")
    for b_p in range(num_p):
        date_p_b_p = int(date_p[b_p])                       # date
        x_b_p = x_point_p[b_p, 0]                           # x
        area_b_p = int(x_point_p[b_p, 1])                   # area
        step_b_p = int(x_point_p[b_p, 2])                   # bench
        infor_p[b_p, 0] = date_p_b_p / 146                  # Date
        infor_p[b_p, 1] = Conditions[date_p_b_p, int(area_b_p +step_b_p*3)]/100
                                                            # Excavation_position
        infor_p[b_p, 2] = Conditions[date_p_b_p, int(area_b_p +step_b_p*3 +6)]/100
                                                            # Struct_position

        if point_or_boundary == 'points':                   # Position X
            infor_p[b_p, 3] = x_b_p/100                     # Measuring Point position
        elif point_or_boundary == 'B':
            infor_p[b_p, 3] = Conditions[date_p_b_p, int(area_b_p +step_b_p*3 +6)]/100
                                                            # boundray B = Struct_position
        elif point_or_boundary == 'C':
            infor_p[b_p, 3] = Conditions[date_p_b_p, int(area_b_p +step_b_p*3)]/100
                                                            # boundray C = Excavation_position
        elif (point_or_boundary == '0') or (point_or_boundary == '100'):
            infor_p[b_p, 3] = int(point_or_boundary)/100    # Well boundary = 0 or 1

        if area_b_p == 0:                                   # Position Y
            infor_p[b_p, 4] = 1.0                           # In Area A
        elif area_b_p == 1:
            infor_p[b_p, 4] = 0                             # In Area B
        elif area_b_p == 2:
            infor_p[b_p, 4] = 0.5                           # In Area C

        infor_p[b_p, 5] = float(step_b_p)                   # Position Z: 0, 1

    infor_p = torch.from_numpy(infor_p).float().to(device)
    return infor_p

def deformation_equations(paras_d, conds_d, segm_d):
    x = conds_d[3]
    if segm_d == 'AB':
        AB_0_parameter, AB_1_parameter, AB_2_parameter, AB_3_parameter, AB_4_parameter, lambda_s = paras_d
        y_equ = (AB_0_parameter
                + torch.exp(lambda_s * x) * (AB_1_parameter * torch.cos(lambda_s * x)
                                            + AB_2_parameter * torch.sin(lambda_s * x))
                + torch.exp(- lambda_s * x) * (AB_3_parameter * torch.cos(lambda_s * x)
                                            + AB_4_parameter * torch.sin(lambda_s * x))
                )
    elif segm_d == 'BC':
        BC_0_parameter, BC_1_parameter, BC_2_parameter, BC_3_parameter, BC_4_parameter, blank_para = paras_d
        y_equ = (BC_0_parameter * x ** 4 + BC_1_parameter * x ** 3
                 + BC_2_parameter * x ** 2 + BC_3_parameter * x ** 1
                 + BC_4_parameter
                 )
    elif segm_d == 'CD':
        CD_0_parameter, CD_1_parameter, CD_2_parameter, CD_3_parameter, CD_4_parameter, lambda_f = paras_d
        y_equ = (CD_0_parameter
                 + torch.exp(lambda_f * x) * (CD_1_parameter * torch.cos(lambda_f * x)
                                              + CD_2_parameter * torch.sin(lambda_f * x))
                 + torch.exp(- lambda_f * x) * (CD_3_parameter * torch.cos(lambda_f * x)
                                                + CD_4_parameter * torch.sin(lambda_f * x))
                 )
    elif segm_d == 'undistinguished':
        if (x >= conds_d[2]) and (x <= conds_d[1]):
            BC_0_parameter, BC_1_parameter, BC_2_parameter, BC_3_parameter, BC_4_parameter, blank_para = paras_d
            y_equ = (BC_0_parameter * x ** 4 + BC_1_parameter * x ** 3
                          + BC_2_parameter * x ** 2 + BC_3_parameter * x ** 1
                          + BC_4_parameter
                          )
        elif x < conds_d[2]:
            AB_0_parameter, AB_1_parameter, AB_2_parameter, AB_3_parameter, AB_4_parameter, lambda_s = paras_d
            y_equ = (AB_0_parameter
                          + torch.exp(lambda_s * x) * (AB_1_parameter * torch.cos(lambda_s * x)
                                                       + AB_2_parameter * torch.sin(lambda_s * x))
                          + torch.exp(- lambda_s * x) * (AB_3_parameter * torch.cos(lambda_s * x)
                                                         + AB_4_parameter * torch.sin(lambda_s * x))
                          )
        elif x > conds_d[1]:
            CD_0_parameter, CD_1_parameter, CD_2_parameter, CD_3_parameter, CD_4_parameter, lambda_f = paras_d
            y_equ = (CD_0_parameter
                          + torch.exp(lambda_f * x) * (CD_1_parameter * torch.cos(lambda_f * x)
                                                       + CD_2_parameter * torch.sin(lambda_f * x))
                          + torch.exp(- lambda_f * x) * (CD_3_parameter * torch.cos(lambda_f * x)
                                                         + CD_4_parameter * torch.sin(lambda_f * x))
                          )
    return y_equ.float()

def xita_equations(paras_x, conds_x, segment_x):
    x = conds_x[3]

    if segment_x == 'BC':
        BC_0_parameter, BC_1_parameter, BC_2_parameter, BC_3_parameter, BC_4_parameter, blank_para = paras_x

        xita_pred  = (4 * BC_0_parameter * x ** 3 + 3 * BC_1_parameter * x ** 2
                    + 2 * BC_2_parameter * x ** 1 + BC_3_parameter
                    )
    elif segment_x == 'AB':
        AB_0_parameter, AB_1_parameter, AB_2_parameter, AB_3_parameter, AB_4_parameter, lambda_s = paras_x
        xita_pred  = (   lambda_s * torch.exp(lambda_s * x) *
                      ( (  AB_1_parameter + AB_2_parameter) * torch.cos(lambda_s * x)
                      + (- AB_1_parameter + AB_2_parameter) * torch.sin(lambda_s * x))
                    +  lambda_s * torch.exp(- lambda_s * x) *
                      ( (- AB_3_parameter + AB_4_parameter) * torch.cos(lambda_s * x)
                      + (- AB_3_parameter - AB_4_parameter) * torch.sin(lambda_s * x))
                    )
    elif segment_x == 'CD':
        CD_0_parameter, CD_1_parameter, CD_2_parameter, CD_3_parameter, CD_4_parameter, lambda_f = paras_x
        xita_pred  = ( lambda_f * torch.exp(lambda_f * x) *
                      ( ( CD_1_parameter + CD_2_parameter) * torch.cos(lambda_f * x)
                     + (- CD_1_parameter + CD_2_parameter) * torch.sin(lambda_f * x))
                    + torch.exp(- lambda_f * x) *
                     ( (- CD_3_parameter + CD_4_parameter) * torch.cos(lambda_f * x)
                     + (- CD_3_parameter - CD_4_parameter) * torch.sin(lambda_f * x))
                    )
    elif segment_x == 'undistinguished':
        if (x >= conds_x[2]) and (x <= conds_x[1]):
            BC_0_parameter, BC_1_parameter, BC_2_parameter, BC_3_parameter, BC_4_parameter, blank_para = paras_x
            xita_pred = (4 * BC_0_parameter * x ** 3 + 3 * BC_1_parameter * x ** 2
                         + 2 * BC_2_parameter * x ** 1 + BC_3_parameter
                         )
        elif x < conds_x[2]:
            AB_0_parameter, AB_1_parameter, AB_2_parameter, AB_3_parameter, AB_4_parameter, lambda_s = paras_x
            xita_pred = (lambda_s * torch.exp(lambda_s * x) *
                         ((AB_1_parameter + AB_2_parameter) * torch.cos(lambda_s * x)
                          + (- AB_1_parameter + AB_2_parameter) * torch.sin(lambda_s * x))
                         + lambda_s * torch.exp(- lambda_s * x) *
                         ((- AB_3_parameter + AB_4_parameter) * torch.cos(lambda_s * x)
                          + (- AB_3_parameter - AB_4_parameter) * torch.sin(lambda_s * x))
                         )
        elif x > conds_x[1]:
            CD_0_parameter, CD_1_parameter, CD_2_parameter, CD_3_parameter, CD_4_parameter, lambda_f = paras_x
            xita_pred = (lambda_f * torch.exp(lambda_f * x) *
                         ((CD_1_parameter + CD_2_parameter) * torch.cos(lambda_f * x)
                          + (- CD_1_parameter + CD_2_parameter) * torch.sin(lambda_f * x))
                         + torch.exp(- lambda_f * x) *
                         ((- CD_3_parameter + CD_4_parameter) * torch.cos(lambda_f * x)
                          + (- CD_3_parameter - CD_4_parameter) * torch.sin(lambda_f * x))
                         )
    return xita_pred.float()

def compound_deformation(model_modi, initial_parameter_cd,
                         x_cd, date_cd):
    num_cd = date_cd.shape[0]
    y_cd = np.empty((num_cd, 1), dtype="float32")
    y_cd = torch.from_numpy(y_cd).to(device)

    major_pd_infor = processing_point_date_information(x_cd, date_cd, 'points')
    major_parameter_modify = model_modi(initial_parameter_cd, major_pd_infor,'undistinguished')

    for i1_cd in range(num_cd):
        y_cd[i1_cd, 0] = deformation_equations(
                                            major_parameter_modify[i1_cd],
                                            major_pd_infor[i1_cd],'undistinguished')

    for a_cd in range(2):
        x_a_cd = x_cd.clone()
        for i2_cd in range(num_cd):
            date_i_cd = int(date_cd[i2_cd])
            area_i_cd = (int(x_cd[i2_cd, 1]) + a_cd + 1) % 3
            step_i_cd = 0
            if (Area_Gantt_chart[date_i_cd, area_i_cd+3] == 1
                ) or (Area_Gantt_chart[int(date_i_cd - 20), area_i_cd+3] == 1):
                step_i_cd = 1
            x_a_cd[i2_cd, 1] = area_i_cd
            x_a_cd[i2_cd, 2] = step_i_cd

        minor_pd_infor_a = processing_point_date_information(x_a_cd, date_cd, 'points')
        minor_parameter_modify = model_modi(initial_parameter_cd, minor_pd_infor_a, 'undistinguished')

        for i3_cd in range(num_cd):
            relative_distance = torch.abs(minor_pd_infor_a[i3_cd, 4]-major_pd_infor[i3_cd, 4])
            relative_distance = relative_distance.cpu().detach().numpy()
            if relative_distance == 0.5:
                y_cd[i3_cd, 0] = y_cd[i3_cd, 0] + (
                                deformation_equations(minor_parameter_modify[i3_cd],
                                                  minor_pd_infor_a[i3_cd],'undistinguished')
                                * neighbor_impact)
    return y_cd

def boundary_loss_calculation(model_modi, initial_parameters_blc,
                              x_blc, date_blc, boundconstr):
    y_boundconstr_loss = 0
    xita_loss = 0
    num_blc = date_blc.shape[0]
    infor_boundconstr = processing_point_date_information(x_blc, date_blc, boundconstr)

    if (boundconstr == '0') or (boundconstr == '100'):
        paras_boundconstr = model_modi(initial_parameters_blc, infor_boundconstr, 'undistinguished')
        for b1_bc in range(num_blc):
            y_boundconstr_loss = (y_boundconstr_loss + torch.abs(
                deformation_equations(paras_boundconstr[b1_bc], infor_boundconstr[b1_bc],
                                      'undistinguished') - 0)
            )
            xita_loss = xita_loss + torch.abs(
                xita_equations(paras_boundconstr[b1_bc], infor_boundconstr[b1_bc],
                                      'undistinguished') - 0
            )
    elif boundconstr == 'B':
        paras_boundconstr_AB = model_modi(initial_parameters_blc, infor_boundconstr, 'AB')
        paras_boundconstr_BC = model_modi(initial_parameters_blc, infor_boundconstr, 'BC')
        for b2_bc in range(num_blc):
            y_boundconstr_loss = y_boundconstr_loss + torch.abs(
                deformation_equations(paras_boundconstr_AB[b2_bc], infor_boundconstr[b2_bc], 'AB')
                - deformation_equations(paras_boundconstr_BC[b2_bc], infor_boundconstr[b2_bc], 'BC')
            )
            xita_loss = xita_loss + torch.abs(
                xita_equations(paras_boundconstr_AB[b2_bc], infor_boundconstr[b2_bc], 'AB')
                - xita_equations(paras_boundconstr_BC[b2_bc], infor_boundconstr[b2_bc], 'BC')
            )
    elif boundconstr == 'C':
        paras_boundconstr_BC = model_modi(initial_parameters_blc, infor_boundconstr, 'BC')
        paras_boundconstr_CD = model_modi(initial_parameters_blc, infor_boundconstr, 'CD')
        for b3_bc in range(num_blc):
            y_boundconstr_loss = y_boundconstr_loss + torch.abs(
                deformation_equations(paras_boundconstr_BC[b3_bc], infor_boundconstr[b3_bc], 'BC')
                - deformation_equations(paras_boundconstr_CD[b3_bc], infor_boundconstr[b3_bc], 'CD')
            )
            xita_loss = xita_loss + torch.abs(
                xita_equations(paras_boundconstr_BC[b3_bc], infor_boundconstr[b3_bc], 'BC')
                - xita_equations(paras_boundconstr_CD[b3_bc], infor_boundconstr[b3_bc], 'CD')
            )

    loss = (y_boundconstr_loss + xita_loss) / num_blc
    return loss

def PI_loss_calculation(model_modi, para, x_PIlc, d_PIlc):
    Words=['0', '100', 'B', 'C']
    l_PIlc = 0
    for w in Words:
        l_PIlc = l_PIlc + boundary_loss_calculation(model_modi, para,
                                                    x_PIlc, d_PIlc, w)
    return (l_PIlc / 8)

def point_fitting_loss_calculation(model_modi, initial_parameters_pflc, x_pflc, date_pflc):
    num_pflc = date_pflc.shape[0]
    loss1 = 0

    for oreder_f in range(6):
        x_i_f = x_pflc.clone()
        for b1_f in range(num_pflc):
            date_b1_f = int(date_pflc[b1_f])
            area_b1_f = int(x_pflc[b1_f, 1])
            step_b1_f = int(x_pflc[b1_f, 2])

            if ((Conditions[date_b1_f, int(area_b1_f +step_b1_f*3 +6)]
                 - 2) // 6) < 2:
                x_i_f[b1_f, 0] = (2 + 1 * 6 + oreder_f * 6)
            elif ((Conditions[date_b1_f, int(area_b1_f +step_b1_f*3 +6)]
                 - 2) // 6) >= 12:
                x_i_f[b1_f, 0] = (2 + 10 * 6 + oreder_f * 6)
            else:
                x_i_f[b1_f, 0] = (2
                                   + (((Conditions[date_b1_f, int(area_b1_f +step_b1_f*3 +6)]
                                        - 2) // 6) - 1) * 6
                                   + oreder_f * 6)
        y_i_f = compound_deformation(model_modi, initial_parameters_pflc, x_i_f, date_pflc)

        for b2_f in range(num_pflc):
            date_b2_f = int(date_pflc[b2_f])
            area_b2_f = int(x_pflc[b2_f, 1])
            point_b2_f = int((x_i_f[b2_f, 0] - 2) / 6)
            loss1 = loss1 + torch.abs(y_i_f[b2_f, 0]
                    - Deformations[date_b2_f, (point_b2_f - 1 + area_b2_f * 15)]
            )

    loss2 = 0
    for order_b in range(6):
        x_i_b = x_pflc.clone()
        mearsure_y_i_b = torch.zeros(num_pflc)
        for b1_b in range(num_pflc):
            date_b1_b = int(date_pflc[b1_b])
            area_b1_b = int(x_pflc[b1_b, 1])
            x_i_b[b1_b, 0] = X_BC[date_b1_b, area_b1_b, order_b]
            mearsure_y_i_b[b1_b] = Deformations_BC[date_b1_b, area_b1_b, order_b]

        y_i_b = compound_deformation(model_modi, initial_parameters_pflc, x_i_b, date_pflc)
        for b2_b in range(num_pflc):
            loss2 = loss2 + torch.abs(y_i_b[b2_b, 0] - mearsure_y_i_b[b2_b])

    return (loss1 + loss2) / num_pflc

def train(model_para, model_modi):

    optimizer_para = torch.optim.Adam(model_para.parameters(),
                                     lr=learning_rate
                                     )
    optimizer_modi = torch.optim.Adam(model_modi.parameters(),
                                     lr=learning_rate
                                     )
    loss_func = nn.L1Loss(reduction='mean')

    for ep in range(epoch):
        for i_train, (X_train, Date_train, Y_measured_train) in enumerate(Input_train_dataloader):

            Y_measured_train = Y_measured_train.float().to(device)
            Inputdata_train = choose_data(X_train, Date_train, 'train')

            optimizer_para.zero_grad()
            optimizer_modi.zero_grad()

            Parameters_initial_train = model_para(Inputdata_train)

            Y_pred_train = compound_deformation(model_modi, Parameters_initial_train,
                                                X_train, Date_train)

            loss_PI = PI_loss_calculation(model_modi, Parameters_initial_train,
                                          X_train, Date_train)

            loss_point_fitting = point_fitting_loss_calculation(
                                            model_modi, Parameters_initial_train,
                                            X_train, Date_train)
            loss_latest_MAE = loss_func(Y_measured_train, Y_pred_train)

            loss_weighted = (
                            (loss_latest_MAE + loss_point_fitting) * 0.7 / 13
                            + loss_PI * 0.3
                            )
            loss_weighted.backward()
            optimizer_para.step()
            optimizer_modi.step()
        print('epoch={}/{}'.format(ep, epoch))
    return


def test(model_para, model_modi):
    model_para.eval()
    model_modi.eval()

    Y_sum_test = np.empty((1, future_dates), dtype="float32")
    Y_pred_sum_test = torch.from_numpy(Y_sum_test).to(device)
    Y_measured_sum_test = torch.from_numpy(Y_sum_test).to(device)

    for i_test, (X_test, Date_test, Y_measured_test) in enumerate(Input_test_dataloader):
        Y_measured_test = Y_measured_test.float().to(device)
        Inputdata_test = choose_data(X_test, Date_test, 'test')

        with torch.no_grad():
            Parameters_initial_test = model_para(Inputdata_test)
            y_pred_test = compound_deformation(model_modi, Parameters_initial_test,
                                                X_test, Date_test)
            for f_test in range(future_dates):
                Y_pred_sum_test[i_test, f_test] = y_pred_test[f_test, 0]
                Y_measured_sum_test[i_test, f_test] = Y_measured_test[f_test]

    Y_pred_test_np = Y_pred_sum_test.cpu().data.numpy()
    print(Y_pred_test_np)
    return

if __name__ == '__main__':

    epoch = 100
    learning_rate = 0.001
    hidden_size_for_models = 32
    batch_size = 16
    timesteps = 3
    num_for_fit_curve = 200
    neighbor_impact = 0.1
    train_times = 1
    Area = 0    #  A：0， B：1， C：2
    Point = 7   #  Point values 0-16
    history_dates = 15
    future_dates = 3

    Area_Gantt_chart = np.zeros((21,6), dtype=int)
    Area_Gantt_chart[ 0 : 20, 0] = 1
    Area_Gantt_chart[12 : 20, 1] = 1

    Input_train = LoadExcelDataset(0, history_dates, Area_Gantt_chart,
                                   'train', 'all', 'all'
                                    )
    Input_test = LoadExcelDataset(int(history_dates), future_dates,  Area_Gantt_chart,
                                  'test', Point, Area
                                )

    Input_train_dataloader = DataLoader(Input_train, batch_size=batch_size,
                                        shuffle=True, num_workers=0, drop_last=True)
    Input_test_dataloader = DataLoader(Input_test, batch_size=future_dates,
                                      shuffle=False, num_workers=0, drop_last=True)

    Items_Conds, _ = readmyexcel('./Input_condition+items.xlsx',
                                'Sheet4', 'B:FF')
    Conditions, _ = readmyexcel('./Input_condition+items.xlsx',
                                'Sheet4', 'B:M')
    Deformations, _ = readmyexcel('./Deformation.xlsx',
                                'Sheet1(15)', 'B:AT')
    X_BC, Deformations_BC = BC_point_multiplication(Deformations, Conditions)

    for jishu in range(train_times):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        model_para = LSTM_Att(input_dim=int(3 + Items_Conds.shape[1]),
                              hidden_size=hidden_size_for_models)
        model_para.to(device)
        model_modi = Parameter_Modification(hidden_size=hidden_size_for_models)
        model_modi.to(device)

        train(model_para, model_modi)
        test(model_para, model_modi)

        del model_para
        del model_modi
        torch.cuda.empty_cache()
