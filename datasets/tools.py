import math
import torch
import torch.nn.functional as F
import random
import numpy as np
from math import sin, cos
from torchvision import transforms
from torchvision import transforms as transform_lib

import scipy.interpolate as si


def aug_look(name):
    if 'selectFrames' in name:
        return SelectFrames(selected_frames)
    elif 'warp' in name:
        return Warp()
    elif 'subtract' in name:
        return Subtract()
    elif 'temporalFlip' in name:
        return TemporalFlip()  # subSampleFlip(data_numpy, time_range)
    elif 'spatialFlip' in name:
        return SpatialFlip()  # subSampleFlip(data_numpy, time_range)
    elif 'rotate' in name:
        return Rotate()  # rotate(data_numpy, axis, angle)
    elif 'center' in name:
        return Center()
    # elif 'zeroOutJoints' in name:
    #     return Zero_out_joints()  # zero_out_joints(data_numpy, joint_list, time_range)
    elif 'shear' in name:
        return Shear()
    else:
        raise IndentationError("Wrong transformation name!")


# def get_transforms(aug_name, selected_frames):
#     aug_name_list = aug_name.split("_")
#     transform_aug = [aug_look('selectFrames', selected_frames)]

#     if aug_name_list[0] != 'None':
#         for i, aug in enumerate(aug_name_list):
#             transform_aug.append(aug_look(aug, selected_frames=None))
            
#     transform_aug.extend([ToTensor(), ])
#     transform_aug = transforms.Compose(transform_aug)

#     return transform_aug

def get_transforms(aug_name):
    aug_name_list = aug_name.split("_")
    # transform_aug = [aug_look('selectFrames', selected_frames)]
    transform_aug = []
    if aug_name_list[0] != 'None':
        for i, aug in enumerate(aug_name_list):
            transform_aug.append(aug_look(aug))
            
    # transform_aug.extend([ToTensor(), ])
    transform_aug = transforms.Compose(transform_aug)

    return transform_aug


class SelectFrames(object):
    def __init__(self, frames):
        self.frames = frames

    def __call__(self, data_numpy):
        return data_numpy[:self.frames, :, :, :]


class ToTensor(object):
    def __call__(self, data_numpy):
        return torch.from_numpy(data_numpy)


class Center(object):
    def __init__(self):
        self.joint = 1

    def __call__(self, data_numpy):
        mean = data_numpy.mean(2)
        if data_numpy.shape[1]==2 and data_numpy[:,1].mean()!=0:
            mean = mean.mean((0,1))
            x_new = data_numpy - mean[None, None, None, :]
        
        else:
            mean = mean.mean(0)
            x_new = data_numpy - mean[None, :, None, :]
        
        return x_new

class Subtract(object):
    def __init__(self):
        self.joint = 1

    def __call__(self, data_numpy):
        x_new = data_numpy.copy()
        # for i in range(data_numpy.shape[2]):
        #     x_new[:, :, i, :] = data_numpy[:, :, i, :] - data_numpy[:, :, self.joint, :]
        x_new = data_numpy - data_numpy[:, :, self.joint, :][:,:,None,:]
        
        return x_new


class TemporalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_numpy):
        if random.random() < self.p:
            # time_range_order = [i for i in range(data_numpy.shape[1])]
            # time_range_reverse = list(reversed(time_range_order))
            # return data_numpy[:, time_range_reverse, :, :]
            data_numpy = data_numpy[::-1,:,:,:]
        return data_numpy.copy()

class SpatialFlip(object):
    def __init__(self, p=0.5, axis=[0,1]):
        self.p = p
        self.axis = axis

    def __call__(self, data_numpy):
        if random.random() < self.p:
            # time_range_order = [i for i in range(data_numpy.shape[1])]
            # time_range_reverse = list(reversed(time_range_order))
            # return data_numpy[:, time_range_reverse, :, :]
            axis = np.random.choice(self.axis)
            data_numpy[:,:,:,axis] = data_numpy[:,:,:,axis].mean() + (data_numpy[:,:,:,axis].mean() - data_numpy[:,:,:,axis]) 
            
        return data_numpy.copy()

class Rotate(object):
    def __init__(self, axis=2, angle=180):
        self.first_axis = axis
        self.first_angle = angle

    def __call__(self, data_numpy):
        axis_next = random.randint(0, 2) if self.first_axis is None else self.first_axis

        angle_next = random.uniform(-self.first_angle, +self.first_angle)

        # temp = data_numpy.copy()
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
        data_numpy = np.dot(data_numpy, R)
        # temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        # temp = temp.transpose(3, 0, 1, 2)
        return data_numpy

class Rotate_T(object):
    def __init__(self, axis=2, angle=180):
        self.first_axis = axis
        self.first_angle = angle

    def __call__(self, data_numpy):
        # axis_next = random.randint(0, 2) if self.first_axis is None else self.first_axis
        axis_next = self.first_axis
        # angle_next = random.uniform(-self.first_angle, +self.first_angle)

        # temp = data_numpy.copy()
        angle = math.radians(self.first_angle)
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
        data_numpy = np.dot(data_numpy, R)
        # temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        # temp = temp.transpose(3, 0, 1, 2)
        return data_numpy


# class Warp(object):
#     def __init__(self):
#         self.sigma = 4

#     def __call__(self, data_numpy):

#         temp = data_numpy.copy()# [3, 150, 25, 2]

#         # Count number of non-zeros frames
#         norm0 = np.linalg.norm(temp[:, :, :, 0].reshape((150, -1)), axis=1)
#         cnt0 = np.count_nonzero(norm0)
#         norm1 = np.linalg.norm(temp[:, :, :, 1].reshape((150, -1)), axis=1)
#         cnt1 = np.count_nonzero(norm1)

#         # Generate rate re-parametrization function
#         gama = generate_gama(cnt0, self.sigma)

#         # Apply time warping transformation
#         temp[:, :cnt0, :, 0] = linear_resampling(temp[:, :cnt0, :, 0], gama, cnt0)

#         threshold = 20  # To make sure sequence of second actor doesn't contain noise only
#         if cnt1 > threshold:
#             temp[:, :cnt0, :, 1] = linear_resampling(temp[:, :cnt0, :, 1], gama, cnt0)

#         return temp

# class Interpolate(object):
#     def __init__(self):
#         self.sigma = 4

#     def __call__(self, data_numpy):
        
#         t, p, j, c = data_numpy.shape
        
#         f = si.interp1d(np.linspace(0,1,t), data_numpy, axis=0)

#         T = np.random.choice(270) + 30 # min 30 max 300
#         new_data = f(np.linspace(0,1,T))
        
#         return new_data

class Interpolate(object):
    def __init__(self, min_len, max_len=300, probs=None):
        self.min_len = min_len
        self.max_len = max_len
        self.probs = probs

    def __call__(self, data):
        
        t, p, j ,c = data.shape
        
        # f = si.interp1d(np.linspace(0,1,t), data_numpy, axis=0)
        if self.probs is None:
            T = np.random.choice(self.max_len-self.min_len) + self.min_len # min 30 max 300
        else:
            T = np.random.choice(len(self.probs), p=self.probs) + self.min_len # min 30 max 300

        # new_shape = [np.random.choice(270) + 30] + list(shape[1:])
        # new_shape = [np.random.choice(270) + 30, shape[3]]
        # new_data = F.interpolate(data.permute([2,1,0,3]), size=new_shape).permute([2,1,0,3])
        new_data = si.interp1d(np.linspace(0,1,t), data, axis=0)(np.linspace(0,1,T))
        # print(data.shape, new_data.shape)
        return new_data



class Warp(object):
    def __init__(self):
        self.sigma = 4

    def __call__(self, data_numpy):
        
        T, p, j, c = data_numpy.shape
        
        new_data = data_numpy

        # Generate rate re-parametrization function
        gama = generate_gama(T, self.sigma)[0]
        # print(gama)

        # Apply time warping transformation
        new_data[:, 0, :, :] = linear_resampling(new_data[:, 0, :, :], gama, T)
        # print(new_data[:, 0, :, :])

        threshold = 20  # To make sure sequence of second actor doesn't contain noise only
        if p == 2 and data_numpy[:,1].sum()!=0:
            # new_data[:, 0, :, :] = linear_resampling(new_data[:, 0, :, :], gama, T)
            new_data[:, 1, :, :] = linear_resampling(new_data[:, 1, :, :], gama, T)

        return new_data

# from datasets import tools
# from importlib import reload
# reload(tools)
# # a = np.arange(10)[:,None,None,None].astype(float)
# a = np.random.normal(0,1,[30,1,25,3])
# a = np.concatenate([a]*3, 3)
# warp = tools.Warp()
# b=warp(a)
# print(b)

class Zero_out_joints(object):
    def __init__(self, joint_list = None, time_range = None):
        self.first_joint_list = joint_list
        self.first_time_range = time_range

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape

        random_int = random.randint(5, 15)
        all_joints = [i for i in range(V)]
        joint_list_ = random.sample(all_joints, random_int)
        joint_list_ = sorted(joint_list_)

        if T < 100:
            random_int = random.randint(20, 50)
        else:
            random_int = random.randint(50, 100)
        all_frames = [i for i in range(T)]
        time_range_ = random.sample(all_frames, random_int)
        time_range_ = sorted(time_range_)

        x_new = np.zeros((C, len(time_range_), len(joint_list_), M))

        temp2 = temp[:, time_range_, :, :].copy()
        temp2[:, :, joint_list_, :] = x_new
        temp[:, time_range_, :, :] = temp2
        return temp


class Shear(object):
    def __init__(self, s1 = None, s2 = None):
        
        self.s1 = s1
        self.s2 = s2

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        s1_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

        s2_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

        R = np.array([[1,     s1_list[0], s2_list[0]],
                      [s1_list[1], 1,     s2_list[1]],
                      [s1_list[2], s2_list[2], 1]])

        R = R.transpose()
        
        # temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        # temp = temp.transpose(3, 0, 1, 2)

        temp = np.dot(temp, R)

        return temp


class Shear_T(object):
    def __init__(self, s1 = None, s2 = None):
        
        self.s1 = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] if s1 is None else s1
        self.s2 = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] if s2 is None else s2

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        
        
        R = np.array([[1,     self.s1[0], self.s2[0]],
                      [self.s1[1], 1,     self.s2[1]],
                      [self.s1[2], self.s2[2], 1]])
        R = R.transpose()        
        temp = np.dot(temp, R)

        return temp


# class Shear(object):
#     def __init__(self, s1 = None, s2 = None):
#         self.s1 = s1
#         self.s2 = s2

#     def __call__(self, data_numpy):
#         temp = data_numpy.copy()
#         s1_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

#         s2_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

#         R = np.array([[1,     s1_list[0], s2_list[0]],
#                       [s1_list[1], 1,     s2_list[1]],
#                       [s1_list[2], s2_list[2], 1]])

#         R = R.transpose()
        
#         temp = np.dot(temp, R)

#         return temp


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

    psi_ = psi*psi
    psi_ = psi_/psi_.sum()
    gama[0, 1:length] = np.cumsum(psi_)
    # gama[0, 1:length] = np.cumsum(np.multiply(psi, psi)) / length

    return gama


def linear_resampling(original_sequence, s, T):

    ##########
    original_sequence = original_sequence.transpose([2,0,1])
    ##########

    C, m1, V = original_sequence.shape  # [3, 150, 25]
    C, m1, V = original_sequence.shape  # [3, 150, 25]

    final_sequence = np.zeros((C, T, V))
    sequence = np.zeros((C*V, m1))

    sequence[0:V, :] = np.transpose(original_sequence[0, :, :])
    sequence[V:V*2, :] = np.transpose(original_sequence[1, :, :])
    sequence[V*2:V*3, :] = np.transpose(original_sequence[2, :, :])

    tau = np.array([((i + 1) / (m1 - 1) - 1 / (m1 - 1)) for i in range(m1)])

    # Re-sampling
    sequence_res = np.zeros((V*C, s.shape[0]))
    for i in range(s.shape[0]):
        k = np.where(s[i] <= tau[:])
        if k[0].shape[0] == 0:
            ind1 = -2
            ind2 = -1
        else:
            ind1 = k[0][0]-1
            ind2 = k[0][0]
            if ind1 == -1:
                ind1 = 0
                ind2 = 1

        w1 = (s[i]-tau[ind1])/(tau[ind2]-tau[ind1])
        w2 = (tau[ind2] - s[i]) / (tau[ind2] - tau[ind1])

        x_new = sequence[:, ind1]
        y_new = sequence[:, ind2]


        theta = np.linalg.norm(x_new - y_new)
        
        if theta > 0:
            sequence_res[:, i] = (1 / theta) * (np.linalg.norm(w2 * theta) * x_new + np.linalg.norm(w1 * theta) * y_new)
        else:
            sequence_res[:, i] = y_new

        final_sequence[0, :, :] = np.transpose(sequence_res[0:V, :])
        final_sequence[1, :, :] = np.transpose(sequence_res[V:V*2, :])
        final_sequence[2, :, :] = np.transpose(sequence_res[V*2:V*3, :])

    ##########
    final_sequence = final_sequence.transpose([1,2,0])
    ##########

    return final_sequence



class RandomSpatialFlip(SpatialFlip):
    def __init__(self, prob=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob
    
    def __call__(self, data):
        if np.random.uniform()<self.prob:
            return super().__call__(data)
        else:
            return data

class RandomRotate(Rotate):
    def __init__(self, prob=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob
    
    def __call__(self, data):
        if np.random.uniform()<self.prob:
            return super().__call__(data)
        else:
            return data

class RandomInterpolate(Interpolate):
    def __init__(self, prob=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob
    
    def __call__(self, data):
        if np.random.uniform()<self.prob:
            return super().__call__(data)
        else:
            return data

class RandomTemporalFlip(TemporalFlip):
    def __init__(self, prob=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob
    
    def __call__(self, data):
        if np.random.uniform()<self.prob:
            return super().__call__(data)
        else:
            return data

class RandomShear(Shear):
    def __init__(self, prob=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob
    
    def __call__(self, data):
        if np.random.uniform()<self.prob:
            return super().__call__(data)
        else:
            return data

class RandomWarp(Warp):
    def __init__(self, prob=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob
    
    def __call__(self, data):
        if np.random.uniform()<self.prob:
            return super().__call__(data)
        else:
            return data