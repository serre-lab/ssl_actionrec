import math
import torch
import random
import code
import numpy as np
from torch import nn
from math import sin, cos
import torch.nn.functional as F
from scipy.interpolate import griddata

from torchvision import transforms as transform_lib


def aug_look(name, args1=None, args2=None):
    if 'selectFrames' in name:
        return SelectFrames(args1)
    elif 'warp' in name:
        return Warp(args1)
    elif 'subtract' in name:
        return Subtract(args1)
    elif 'randomFlip' in name:
        return RandomHorizontalFlip()  # subSampleFlip(data_numpy, time_range)
    elif 'zeroOutAxis' in name:
        return Zero_out_axis(args1)  # zero_out_axis(data_numpy, axis)
    elif 'rotate' in name:
        return Rotate(args1, args2)  # rotate(data_numpy, axis, angle)
    elif 'zeroOutJoints' in name:
        return Zero_out_joints(args1, args2)  # zero_out_joints(data_numpy, joint_list, time_range)
    elif 'gausNoise' in name:
        # return Gaus_noise( args1, args2)  # gaus_noise(data_numpy, mean= 0, std = 0.01)
        return Gaus_noise()  # gaus_noise(data_numpy, mean= 0, std = 0.01)
    elif  'gausFilter' in name:
        # return Gaus_filter(args1, args2)  # gaus_filter(data_numpy)
        return Gaus_filter()  # gaus_filter(data_numpy)
    elif 'shear' in name:
        return Shear(args1, args2)
    else:
        raise IndentationError("wrong")


class SelectFrames(object):
    def __init__(self, frames):
        self.frames = frames

    def __call__(self, data_numpy):
        return data_numpy[:, :self.frames, :, :]


class ToTensor(object):
    def __call__(self, data_numpy):
        return torch.from_numpy(data_numpy)


class Subtract(object):
    def __init__(self, joint=None):
        if joint == None:
            self.joint = random.randint(0, 24)
        else:
            self.joint = joint

    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((C, T, V, M))
        # for i in range(V):
        #     x_new[:, :, i, :] = data_numpy[:, :, i, :] - data_numpy[:, :, self.joint, :]
        x_new = data_numpy - data_numpy[:, :, self.joint, :][:,:,None,:]
    
        return x_new


class Warp(object):
    def __init__(self, sigma):
        if sigma == None:
            self.sigma = 4
        else:
            self.sigma = sigma

    def __call__(self, data_numpy):

        C, T, V, M = data_numpy.shape # [3, 150, 25, 2]

        x_new = np.zeros((C, T, V, M))
        # Keep only non-zero frames
        
        # cnt = 0
        # cnt1 = 0
        # for i in range(T):
        #     if np.linalg.norm(data_numpy[:, i, :, 0]) != 0:
        #         cnt += 1
        #     if np.linalg.norm(data_numpy[:, i, :, 1]) != 0:
        #         cnt1 += 1

        norms = np.linalg.norm(data_numpy, axis=(0,2))
        cnt = (norms[:,0]!=0).sum()
        cnt1 = (norms[:,1]!=0).sum()
        
        # generate random number between 150 and 50
        number = random.randint(50, 150)
        s = np.arange(0, 1, 1 / number)
        #s1 = np.arange(0, 1, 1 / cnt)
        if number != len(s):
            s = s[:number]

        #gama = generate_gama(number, self.sigma)
        if cnt > 20:
             # original : gama = generate_gama(cnt, self.sigma)

            #ff = interpolate_sequence(data_numpy[:, :cnt, :, 0], s, s1)
            #x_new[:, :ff.shape[1], :, 0] = ff
            #x_new[:, :cnt, :, 0] = interpolate_sequence(data_numpy[:, :cnt, :, 0], gama, s)
            x_new[:, :number, :, 0] = linear_resampling(data_numpy[:, :cnt, :, 0], s, number)

        if cnt1 > 20:

            #s = np.arange(0, 1, 1 / cnt1)

            #if cnt1 != len(s):
                #s = s[:cnt1]
            #gama = generate_gama(number, self.sigma)
            #ff = interpolate_sequence(data_numpy[:, :cnt, :, 1], s, s1)
            #x_new[:, :ff.shape[1], :, 1] = ff
            #x_new[:, :cnt1, :, 1] = interpolate_sequence(data_numpy[:, :cnt1, :, 1], gama, s)
            x_new[:, :number, :, 1] = linear_resampling(data_numpy[:, :cnt, :, 1], s, number)

        return x_new


def interpolate_sequence(sequence, gam, s):
    d, k, N = sequence.shape
    final_sequence = np.zeros((d, gam.size, N))
    p2 = np.zeros((N * d, k))
    for i in range(k):
        p = sequence[:, i, :]
        for j in range(N):
            p2[j, i] = p[0, j]
            p2[j + N, i] = p[1, j]
            p2[j + 2 * N, i] = p[2, j]

    p2 = np.transpose(p2)

    f = griddata(s, p2, gam, method='nearest')

    #f[:, gam.size-1:, :] = f[:, gam.size-2, :]

    final_sequence[0, :, :] = f[:, 0:25]
    final_sequence[1, :, :] = f[:, 25:50]
    final_sequence[2, :, :] = f[:, 50:75]

    return final_sequence


def generate_gama(length, sigma):
    gama = np.zeros((1, length))
    tt = length - 1
    time = np.linspace(0, 1, tt)
    mu = np.sqrt(np.ones((1, length - 1)) * tt / (length - 1))
    omega = 2*np.pi/tt

    alpha_i = np.random.normal(0, sigma)
    v = alpha_i * np.ones((1, tt))
    cnt = 1
    for l in range(1, 10):
        alpha_i = np.random.normal(0, sigma)
        if l % 2 != 0:
            v += alpha_i * np.sqrt(2)*np.cos(cnt*omega*time)
            cnt += 1
        if l % 2 == 0:
            v += alpha_i * np.sqrt(2) * np.sin(cnt * omega * time)

    v = v - (float(np.matmul(mu, np.transpose(v))))*mu/tt
    vn = np.linalg.norm(v) / np.sqrt(tt)
    psi = np.cos(vn) * mu + np.sin(vn) * v / vn
    gama[0, 1:length] = np.cumsum(np.multiply(psi, psi)) / length

    return gama


def linear_resampling(original_sequence, s, T):

    # original_sequence of dim [3, 150, 25, 1]
    #s = s.reshape((s.shape[1]))
    C, TT, V = original_sequence.shape  # [3, 150, 25]

    final_sequence = np.zeros((C, T, V))

    n1 = V*C
    m1 = TT
    tau = np.zeros((m1, m1))

    sequence = np.zeros((C*V, TT))
    sequence[0:25, :] = np.transpose(original_sequence[0, :, :])
    sequence[25:50, :] = np.transpose(original_sequence[1, :, :])
    sequence[50:75, :] = np.transpose(original_sequence[2, :, :])

    for i in range(m1):
        tau[i, 0] = (i+1)/(m1-1)-1/(m1-1)

    # Re-sampling
    sequence_res = np.zeros((n1, len(s)))

    for i in range(len(s)):
        k = np.where(s[i] <= tau[:])
        ind1 = k[0][0]-1
        ind2 = k[0][0]
        if ind1 == -1:
            ind1 = 0
            ind2 = 1

        w1 = (s[i]-tau[ind1, 0])/(tau[ind2, 0]-tau[ind1, 0])
        w2 = (tau[ind2, 0] - s[i]) / (tau[ind2, 0] - tau[ind1, 0])

        x_new = sequence[:, ind1]
        y_new = sequence[:, ind2]

        theta = np.linalg.norm(x_new - y_new)
        if theta > 0:
            sequence_res[:, i] = (1 / theta) * (np.linalg.norm(w2 * theta) * x_new + np.linalg.norm(w1 * theta) * y_new)
        else:
            sequence_res[:, i] = y_new

        final_sequence[0, :, :] = np.transpose(sequence_res[0:25, :])
        final_sequence[1, :, :] = np.transpose(sequence_res[25:50, :])
        final_sequence[2, :, :] = np.transpose(sequence_res[50:75, :])

    return final_sequence


class Subsample(object):
    def __init__(self, time_range=None):
        self.time_range = time_range

    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        # frames = random.randint(1, T)
        if self.time_range == None:
            self.time_range = random.randint(1, T)
        all_frames = [i for i in range(T)]
        time_range_list = random.sample(all_frames, self.time_range)
        time_range_list.sort()
        x_new = np.zeros((C, T, V, M))
        x_new[:, time_range_list, :, :] = data_numpy[:, time_range_list, :, :]
        return x_new


class Zero_out_axis(object):
    def __init__(self, axis = None):
        self.first_axis = axis

    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0,2)

        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp


class Diff_on_axis(object):
    def __init__(self, axis = None):
        self.first_axis = axis

    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0,2)
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        for t in range(T - 1):
            temp[axis_next, t, :, :] = data_numpy[axis_next, t + 1, :, :] - data_numpy[axis_next, t, :, :]
            temp[axis_next, -1, :, :] = np.zeros((V, M))
        return temp


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        if random.random() < self.p:
            time_range_order = [i for i in range(T)]
            time_range_reverse = list(reversed(time_range_order))
            return data_numpy[:, time_range_reverse, :, :]
        else:
            return data_numpy.copy()


class Rotate(object):
    def __init__(self, axis=None, angle=None, ):
        self.first_axis = axis
        self.first_angle = angle

    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0,2)

        if self.first_angle != None:
            if isinstance(self.first_angle, list):
                angle_big = self.first_angle[0] + self.first_angle[1]
                angle_small = self.first_angle[0] - self.first_angle[1]
                angle_next = random.uniform(angle_small, angle_big)
            else:
                angle_next = self.first_angle
        else:
            # angle_list = [0, 90, 180, 270]
            # angle_next = random.sample(angle_list, 1)[0]
            angle_next = random.uniform(0, 30)

        temp = data_numpy.copy()
        angle = math.radians(angle_next)
        # x
        if axis_next == 0:
            R = np.array([[1, 0, 0],
                          [0, cos(angle), sin(angle)],
                          [0, -sin(angle), cos(angle)]])
        # y
        if axis_next == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                          [0, 1, 0],
                          [sin(angle), 0, cos(angle)]])

        # z
        if axis_next == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                          [-sin(angle), cos(angle), 0],
                          [0, 0, 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp


class Zero_out_joints(object):
    def __init__(self,joint_list = None, time_range = None):
        self.first_joint_list = joint_list
        self.first_time_range = time_range
    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape

        if self.first_joint_list != None:
            if isinstance(self.first_joint_list, int):
                all_joints = [i for i in range(V)]
                joint_list_ = random.sample(all_joints, self.first_joint_list)
                joint_list_ = sorted(joint_list_)
            else:
                joint_list_ = self.first_joint_list
        else:
            random_int =  random.randint(5, 15)
            all_joints = [i for i in range(V)]
            joint_list_ = random.sample(all_joints, random_int)
            joint_list_ = sorted(joint_list_)

        if self.first_time_range != None:
            if isinstance(self.first_time_range, int):
                all_frames = [i for i in range(T)]
                time_range_ = random.sample(all_frames, self.first_time_range)
                time_range_ = sorted(time_range_)
            else:
                time_range_ = self.first_time_range
        else:
            if T < 100:
                random_int = random.randint(20, 50)
            else:
                random_int = random.randint(50, 100)
            all_frames = [i for i in range(T)]
            time_range_ = random.sample(all_frames, random_int)
            time_range_ = sorted(time_range_)

        x_new = np.zeros((C, len(time_range_), len(joint_list_), M))
        # print("data_numpy",data_numpy[:, time_range, joint_list, :].shape)
        temp2 = temp[:, time_range_, :, :].copy()
        temp2[:, :, joint_list_, :] = x_new
        temp[:, time_range_, :, :] = temp2
        return temp


class Gaus_noise(object):
    def __init__(self, mean= 0, std = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(self.mean, self.std, size=(C, T, V, M))
        return temp + noise


class Gaus_filter(object):
    def __init__(self, kernel=15, sig_list=[0.1, 2]):
        self.g = GaussianBlurConv(3, kernel, sig_list)

    def __call__(self, data_numpy):
        return self.g(data_numpy)


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel = 15, sigma=[0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):

        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        # kernel =  kernel.float()
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1)# (3,1,1,5)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        x = torch.from_numpy(x)
        if prob < 0.5:
            x = x.permute(3, 0, 2, 1)# M,C,V,T
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )),   groups=self.channels)
            x = x.permute(1, -1, -2, 0)#C,T,V,M

        return x.numpy()


class Shear(object):
    def __init__(self, s1 = None, s2 = None):
        self.s1 = s1
        self.s2 = s2

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        if self.s1 != None:
            s1_list = self.s1
        else:
            s1_list = [random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]
            # print(s1_list[0])
        if self.s2 != None:
            s2_list = self.s2
        else:
            s2_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

        R = np.array([[1,     s1_list[0], s2_list[0]],
                      [s1_list[1], 1,     s2_list[1]],
                      [s1_list[2], s2_list[2], 1]])

        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp


class Diff(object):
    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((C, T, V, M))
        for t in range(T - 1):
            x_new[:, t, :, :] = data_numpy[:, t + 1, :, :] - data_numpy[:, t, :, :]
        return x_new

'''========================================================'''





# ok
def subtract(data_numpy, joint):
    C, T, V, M = data_numpy.shape
    x_new = np.zeros((C, T, V, M))
    # for i in range(V):
    #     x_new[:, :, i, :] = data_numpy[:, :, i, :] - data_numpy[:, :, joint, :]
    x_new = data_numpy - data_numpy[:, :, joint, :][:,:,None,:]
    return x_new


# ok
# b: crop and resize
def subsample(data_numpy, time_range):
    C, T, V, M = data_numpy.shape
    if isinstance(time_range, int):
        # all_frames = [i for i in range(T)]
        all_frames = np.arange(T)
        time_range = random.sample(all_frames, time_range)
        time_range.sort()
    x_new = np.zeros((C, T, V, M))
    x_new[:, time_range, :, :] = data_numpy[:, time_range, :, :]
    return x_new


# ok
# c: crop,resize (and flip)
def subSampleFlip(data_numpy, time_range):
    C, T, V, M = data_numpy.shape
    assert T >= time_range, "frames longer than data"
    if isinstance(time_range, int):
        # all_frames = [i for i in range(T)]
        all_frames = np.arange(T)
        time_range = random.sample(all_frames, time_range)
        time_range_order = sorted(time_range)
        time_range_reverse =  list(reversed(time_range_order))
    x_new = np.zeros((C, T, V, M))
    x_new[:, time_range_order, :, :] = data_numpy[:, time_range_reverse, :, :]
    return x_new


# ok
# d: color distort.(drop)
def zero_out_axis(data_numpy, axis):
    # x, y, z -> axis : 0,1,2
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    x_new = np.zeros((T, V, M))
    temp[axis] = x_new
    return temp


def diff_on_axis(data_numpy, axis):
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    # for t in range(T - 1):
        # temp[axis, t, :, :] = data_numpy[axis, t+1, :, :] - data_numpy[axis, t, :, :]
    temp[axis] = data_numpy[axis, 1:, :, :] - data_numpy[axis, :-1, :, :]
    temp[axis, -1, :, :] = np.zeros((V, M))
    return temp


def rotate(data_numpy, axis, angle):
    temp = data_numpy.copy()
    angle = math.radians(angle)
    # x
    if axis == 0:
        R = np.array([[1, 0, 0],
                      [0, cos(angle), sin(angle)],
                      [0, -sin(angle), cos(angle)]])
    # y
    if axis == 1:
        R = np.array([[cos(angle), 0, -sin(angle)],
                       [0,1,0],
                       [sin(angle), 0, cos(angle)]])

    # z
    if axis == 2:
        R = np.array([[cos(angle),sin(angle),0],
                       [-sin(angle),cos(angle),0],
                       [0,0,1]])
    R = R.transpose()
    temp = np.dot(temp.transpose([1,2,3,0]),R)
    temp = temp.transpose(3,0,1,2)
    return temp


def zero_out_joints(data_numpy, joint_list, time_range):
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    # print("joint_list" ,joint_list)
    # print("time_range" ,time_range)
    if isinstance(joint_list, int):
        all_joints = [i for i in range(V)]
        joint_list_ = random.sample(all_joints, joint_list)
        joint_list_ = sorted(joint_list_)
    else:
        joint_list_ = joint_list
    if isinstance(time_range, int):
        all_frames = [i for i in range(T)]
        time_range_ = random.sample(all_frames, time_range)
        time_range_ = sorted(time_range_)
    else:
        time_range_ = time_range
    x_new = np.zeros((C, len(time_range_), len(joint_list_), M))
    # print("data_numpy",data_numpy[:, time_range, joint_list, :].shape)
    temp2 = temp[:, time_range_, :, :].copy()
    temp2[:, :, joint_list_, :] = x_new
    temp[:, time_range_, :, :] = temp2
    return temp


def gaus_noise(data_numpy, mean= 0, std = 0.01):
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    noise = np.random.normal(mean, std, size=(C, T, V, M ))
    return temp + noise


