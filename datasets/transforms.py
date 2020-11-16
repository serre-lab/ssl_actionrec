

from datasets.tools import *
import numpy as np

probs = np.array([
        2.72553829e-04, 5.94662900e-04, 6.68995763e-04, 6.93773384e-04, 8.91994351e-04, 1.09021532e-03, 2.05654253e-03, 2.55209495e-03, 2.75031591e-03, 3.44408930e-03, 3.98919696e-03, 4.38563889e-03,
        6.09529473e-03, 7.08639956e-03, 7.55717436e-03, 9.81193786e-03, 9.06860923e-03, 1.09517084e-02, 1.26861419e-02, 1.15711489e-02, 1.27604747e-02, 1.35038034e-02, 1.41727991e-02, 1.39498005e-02, 1.53621249e-02, 1.59567878e-02, 1.80876632e-02, 1.75177779e-02,
        1.73691122e-02, 1.80133304e-02, 1.75425556e-02, 2.10114225e-02, 2.01194281e-02, 1.94752100e-02, 2.02185386e-02, 1.88557695e-02, 1.75177779e-02, 1.79637751e-02, 1.71461136e-02, 1.85584380e-02, 1.72204465e-02, 1.62293417e-02, 1.57585669e-02, 1.63532298e-02,
        1.68487822e-02, 1.34542481e-02, 1.43710201e-02, 1.39993558e-02, 1.28595852e-02, 1.18684804e-02, 1.40489110e-02, 1.37763572e-02, 1.22401447e-02, 1.08773756e-02, 1.01092693e-02, 1.02579350e-02, 9.39071830e-03, 9.58893927e-03, 9.68804975e-03, 1.09269308e-02,
        9.11816447e-03, 9.44027354e-03, 1.06295993e-02, 9.63849451e-03, 7.23506529e-03, 7.23506529e-03, 7.60672960e-03, 7.38373101e-03, 7.01206670e-03, 6.64040239e-03, 7.08639956e-03, 7.40850864e-03, 7.30939815e-03, 6.91295622e-03, 5.97140663e-03, 5.92185138e-03,
        5.77318566e-03, 5.40152135e-03, 4.58385986e-03, 4.38563889e-03, 4.95552417e-03, 4.45997175e-03, 4.08830744e-03, 4.48474937e-03, 4.43519413e-03, 3.81575361e-03, 3.74142075e-03, 2.08132015e-03, 3.34497882e-03, 3.22109071e-03, 3.27064595e-03, 3.56797740e-03,
        3.41931168e-03, 2.89898164e-03, 3.04764736e-03, 2.60165019e-03, 2.75031591e-03, 3.61753264e-03, 3.76619837e-03, 2.82464878e-03, 2.35387398e-03, 2.25476350e-03, 2.03176491e-03, 2.13087539e-03, 1.88309918e-03, 2.20520826e-03, 2.52731733e-03, 1.83354394e-03, 
        1.43710201e-03, 1.56099011e-03, 2.20520826e-03, 2.20520826e-03, 1.46187963e-03, 1.80876632e-03, 2.20520826e-03, 1.83354394e-03, 1.68487822e-03, 1.48665725e-03, 1.95743205e-03, 1.75921108e-03, 1.43710201e-03, 1.51143487e-03, 1.21410342e-03, 9.66327213e-04, 
        1.09021532e-03, 1.48665725e-03, 1.58576773e-03, 1.56099011e-03, 1.21410342e-03, 7.18551005e-04, 1.04066008e-03, 1.56099011e-03, 1.09021532e-03, 8.91994351e-04, 8.91994351e-04, 7.68106246e-04, 7.43328626e-04, 9.16771972e-04, 8.17661488e-04, 4.95552417e-04, 
        7.18551005e-04, 8.67216730e-04, 8.17661488e-04, 7.92883867e-04, 8.42439109e-04, 7.68106246e-04, 4.95552417e-04, 4.70774796e-04, 4.45997175e-04, 3.71664313e-04, 4.70774796e-04, 3.96441934e-04, 4.95552417e-04, 6.93773384e-04, 3.22109071e-04, 3.71664313e-04, 
        4.70774796e-04, 3.22109071e-04, 3.22109071e-04, 5.20330038e-04, 2.97331450e-04, 2.22998588e-04, 2.97331450e-04, 2.72553829e-04, 2.47776209e-04, 3.71664313e-04, 1.73443346e-04, 3.46886692e-04, 2.22998588e-04, 1.98220967e-04, 3.71664313e-04, 2.47776209e-04, 
        4.21219554e-04, 2.72553829e-04, 3.22109071e-04, 3.22109071e-04, 2.22998588e-04, 7.43328626e-05, 2.72553829e-04, 1.48665725e-04, 7.43328626e-05, 1.98220967e-04, 1.73443346e-04, 1.98220967e-04, 1.48665725e-04, 2.22998588e-04, 2.47776209e-05, 1.23888104e-04, 
        7.43328626e-05, 2.47776209e-05, 4.95552417e-05, 2.22998588e-04, 1.98220967e-04, 7.43328626e-05, 2.47776209e-05, 1.73443346e-04, 1.98220967e-04, 1.98220967e-04, 1.98220967e-04, 9.91104834e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 9.91104834e-05, 
        9.91104834e-05, 1.73443346e-04, 1.23888104e-04, 9.91104834e-05, 4.95552417e-05, 2.47776209e-05, 2.47776209e-05, 4.95552417e-05, 9.91104834e-05, 9.91104834e-05, 7.43328626e-05, 4.95552417e-05, 7.43328626e-05, 4.95552417e-05, 1.23888104e-04, 7.43328626e-05,
        7.43328626e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 7.43328626e-05, 4.95552417e-05, 7.43328626e-05, 4.95552417e-05, 2.47776209e-05, 4.95552417e-05, 7.43328626e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05,
        2.47776209e-05, 1.73443346e-04, 2.47776209e-05, 2.47776209e-05, 7.43328626e-05, 4.95552417e-05, 2.47776209e-05, 2.47776209e-05, 7.43328626e-05, 4.95552417e-05, 4.95552417e-05, 2.47776209e-05, 4.95552417e-05, 4.95552417e-05, 2.47776209e-05, 2.47776209e-05,
        2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 2.47776209e-05, 6.19440521e-04,
       ])
       
vals = np.array([ 32.,  33.,  34.,  35.,  36.,  37.,  38.,  39.,  40.,  41.,  42., 43.,  44.,  45.,  46.,  47.,  48.,  49.,  50.,  51.,  52.,  53.,
        54.,  55.,  56.,  57.,  58.,  59.,  60.,  61.,  62.,  63.,  64., 65.,  66.,  67.,  68.,  69.,  70.,  71.,  72.,  73.,  74.,  75.,
        76.,  77.,  78.,  79.,  80.,  81.,  82.,  83.,  84.,  85.,  86., 87.,  88.,  89.,  90.,  91.,  92.,  93.,  94.,  95.,  96.,  97.,
        98.,  99., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 110., 111., 112., 113., 114., 115., 116., 117., 118., 119.,
       120., 121., 122., 123., 124., 125., 126., 127., 128., 129., 130., 131., 132., 133., 134., 135., 136., 137., 138., 139., 140., 141.,
       142., 143., 144., 145., 146., 147., 148., 149., 150., 151., 152., 153., 154., 155., 156., 157., 158., 159., 160., 161., 162., 163.,
       164., 165., 166., 167., 168., 169., 170., 171., 172., 173., 174., 175., 176., 177., 178., 179., 180., 181., 182., 183., 184., 185.,
       186., 187., 188., 189., 190., 191., 192., 193., 194., 195., 196., 197., 198., 199., 200., 201., 202., 203., 204., 205., 206., 207.,
       208., 209., 210., 211., 212., 213., 214., 215., 216., 217., 218., 219., 220., 221., 222., 223., 224., 225., 226., 227., 228., 229.,
       230., 231., 232., 233., 234., 235., 236., 237., 238., 239., 240., 241., 242., 243., 244., 245., 246., 247., 248., 249., 250., 251.,
       252., 253., 254., 255., 256., 257., 258., 259., 260., 261., 262., 263., 264., 265., 266., 267., 268., 269., 270., 271., 272., 273.,
       274., 275., 276., 277., 278., 279., 280., 281., 282., 283., 284., 285., 286., 287., 288., 289., 290., 291., 292., 293., 294., 295.,
       296., 297., 298., 299., 300.]).astype(int)

class NTUMoco2TrainTransforms:
    def __init__(self):
        self.transform = transform_lib.Compose([
            SpatialFlip(),
            Rotate(),
            Interpolate(min_len=31, probs=probs/probs.sum()),
            # Warp(),
            # Subtract(),
            Center(),
            ToTensor(),
            # Interpolate(),
        ])
        self.gt_transform = transform_lib.Compose([
            ToTensor(),
        ])

    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k


class NTUMoco2EvalTransforms:
    def __init__(self):
        # image augmentation functions
        self.transform = transform_lib.Compose([
            SpatialFlip(),
            Rotate(),
            Interpolate(min_len=31, probs=probs/probs.sum()),
            # Warp(),
            # Subtract(),
            Center(),
            ToTensor(),
            # Interpolate(),
        ])
        self.gt_transform = transform_lib.Compose([
            Center(),
            ToTensor(),
        ])
    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k

##############################################################################################################
##############################################################################################################
##############################################################################################################

class NTUMoco2TrainTransformsTS:
    def __init__(self):
        self.transform = transform_lib.Compose([
            # SpatialFlip(),
            # Rotate(),
            # Interpolate(min_len=31, probs=probs/probs.sum()),
            # Warp(),
            # Subtract(),
            TemporalFlip(),
            Shear(),
            Center(),
            ToTensor(),
            # Interpolate(),
        ])
        self.gt_transform = transform_lib.Compose([
            ToTensor(),
        ])

    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k


class NTUMoco2EvalTransformsTS:
    def __init__(self):
        # image augmentation functions
        self.transform = transform_lib.Compose([
            # SpatialFlip(),
            # Rotate(),
            # Interpolate(min_len=31, probs=probs/probs.sum()),
            # Warp(),
            # Subtract(),
            TemporalFlip(),
            Shear(),
            Center(),
            ToTensor(),
            # Interpolate(),
        ])
        self.gt_transform = transform_lib.Compose([
            Center(),
            ToTensor(),
        ])
    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k

##############################################################################################################
##############################################################################################################
##############################################################################################################

class NTUQConTrainTransformsTS:
    def __init__(self):
        self.transform = transform_lib.Compose([
            # SpatialFlip(),
            # Rotate(),
            # Interpolate(min_len=31, probs=probs/probs.sum()),
            # Warp(),
            # Subtract(),
            TemporalFlip(),
            Shear(),
            Center(),
            ToTensor(),
            # Interpolate(),
        ])
        self.gt_transform = transform_lib.Compose([
            ToTensor(),
        ])

    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k

NTUQConEvalTransformsTS = NTUQConTrainTransformsTS

##############################################################################################################
##############################################################################################################
##############################################################################################################

class NTUMocoTDTrainTransforms(NTUMoco2TrainTransforms):
    def __init__(self):
        # super().__init__()

        self.transform = transform_lib.Compose([
            SpatialFlip(),
            Rotate(),
            Interpolate(min_len=31, probs=probs/probs.sum()),
            TemporalFlip(),
            Shear(),
            # Warp(),
            # Subtract(),
            Center(),
            ToTensor(),
            # Interpolate(),
        ])
        self.gt_transform = transform_lib.Compose([
            Center(),
            ToTensor(),
        ])
    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        gt = self.gt_transform(inp)
        return q, k, gt

class NTUMocoTDEvalTransforms(NTUMoco2EvalTransforms):
    def __init__(self):
        # super().__init__()
        
        self.transform = transform_lib.Compose([
            SpatialFlip(),
            Rotate(),
            Interpolate(min_len=31, probs=probs/probs.sum()),
            TemporalFlip(),
            Shear(),
            # Warp(),
            # Subtract(),
            Center(),
            ToTensor(),
            # Interpolate(),
        ])
        self.gt_transform = transform_lib.Compose([
            Center(),
            ToTensor(),
        ])
    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        gt = self.gt_transform(inp)
        return q, k, gt


##############################################################################################################
##############################################################################################################
##############################################################################################################

class NTUMocoTDTrainTransformsTS:
    def __init__(self):
        # super().__init__()

        self.transform = transform_lib.Compose([
            TemporalFlip(),
            Shear(),
            Center(),
            ToTensor(),
        ])
        self.gt_transform = transform_lib.Compose([
            Center(),
            ToTensor(),
        ])
    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        gt = self.gt_transform(inp)
        return q, k, gt

NTUMocoTDEvalTransformsTS = NTUMocoTDTrainTransformsTS

##############################################################################################################
##############################################################################################################
##############################################################################################################

class NTUMoco2TrainTransformsCAE:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self):

        # image augmentation functions
        self.train_transform = transform_lib.Compose([
            Subtract(),
            TemporalFlip(),
            Shear(),
            ToTensor(),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class NTUMoco2EvalTransformsCAE:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self):

        # image augmentation functions
        self.train_transform = transform_lib.Compose([
            Subtract(),
            SpatialFlip(),
            Shear(),
            ToTensor(),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

##############################################################################################################
##############################################################################################################
##############################################################################################################


class UCLAMoco2TrainTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self):

        # image augmentation functions
        self.train_transform = transform_lib.Compose([
            SpatialFlip(),
            Rotate(),
            Interpolate(min_len=11, max_len=50),
            ToTensor(),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

##############################################################################################################
##############################################################################################################
##############################################################################################################


class UCLAMoco2TrainTransformsTS:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self):

        # image augmentation functions
        self.train_transform = transform_lib.Compose([
            TemporalFlip(),
            Shear(),
            ToTensor(),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

UCLAMoco2EvalTransforms = UCLAMoco2TrainTransforms

##############################################################################################################
##############################################################################################################
##############################################################################################################


class UCLATDConTransformsTS:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self):

        # image augmentation functions
        self.transform = transform_lib.Compose([
            TemporalFlip(),
            Shear(),
            ToTensor(),
        ])
        self.gt_transform = transform_lib.Compose([
            ToTensor(),
        ])


    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        gt = self.gt_transform(inp)
        # print(q.shape)
        return q, k, gt

UCLAMocoTDTransformsTS = UCLATDConTransformsTS
# UCLAMoco2EvalTransforms = UCLAMoco2TrainTransforms


##############################################################################################################
##############################################################################################################
##############################################################################################################

class AE_TrainTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self):

        # image augmentation functions
        self.train_transform = transform_lib.Compose([
            ToTensor(),
        ])

    def __call__(self, inp):
        return self.train_transform(inp)

class AE_EvalTransforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self):

        # image augmentation functions
        self.train_transform = transform_lib.Compose([
            ToTensor(),
        ])

    def __call__(self, inp):
        return self.train_transform(inp)


##############################################################################################################
##############################################################################################################
############# ANALYSIS TRANSFORMS ############################################################################
##############################################################################################################
##############################################################################################################

# RandomSpatialFlip
# RandomRotate
# RandomInterpolate
# RandomTemporalFlip
# RandomShear
# RandomWarp

class AN_UCLA_AE_Transforms_1:

    def __init__(self):

        self.transform = transform_lib.Compose([
            RandomSpatialFlip(prob=0.3),
            RandomRotate(prob=0.3),
            RandomShear(prob=0.3),
            RandomTemporalFlip(prob=0.3),
            RandomInterpolate(prob=0.3, min_len=11, max_len=50),
            RandomWarp(prob=0.3),
            ToTensor(),
        ])
        self.gt_transform = transform_lib.Compose([
            ToTensor(),
        ])

    def __call__(self, inp):
        return self.transform(inp)

class AN_UCLA_AE_Transforms_1_gt(AN_UCLA_AE_Transforms_1):

    def __call__(self, inp):
        q = self.transform(inp)
        gt = self.gt_transform(inp)
        return q, gt

class AN_UCLA_AE_Transforms_2(AN_UCLA_AE_Transforms_1):

    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k

class AN_UCLA_AE_Transforms_2_gt(AN_UCLA_AE_Transforms_1):

    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        gt = self.gt_transform(inp)
        return q, k, gt



class AN_UCLA_All(AN_UCLA_AE_Transforms_1):

    def __init__(self):

        self.transform = transform_lib.Compose([
            RandomSpatialFlip(prob=0.3),
            RandomRotate(prob=0.3),
            RandomShear(prob=0.3),
            RandomTemporalFlip(prob=0.3),
            RandomInterpolate(prob=0.3, min_len=11, max_len=50),
            RandomWarp(prob=0.3),
            ToTensor(),
        ])

class AN_UCLA_SpatialFlip(AN_UCLA_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            SpatialFlip(),
            ToTensor(),
        ])

class AN_UCLA_Rotate(AN_UCLA_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            Rotate(),
            ToTensor(),
        ])

class AN_UCLA_Shear(AN_UCLA_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            Shear(),
            ToTensor(),
        ])

class AN_UCLA_TemporalFlip(AN_UCLA_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            TemporalFlip(),
            ToTensor(),
        ])

class AN_UCLA_Interpolate(AN_UCLA_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            Interpolate(min_len=11, max_len=50),
            ToTensor(),
        ])

class AN_UCLA_Warp(AN_UCLA_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            Warp(),
            ToTensor(),
        ])

class AN_UCLA_base(AN_UCLA_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            ToTensor(),
        ])


##############################################################################################################
##############################################################################################################
############# ANALYSIS TRANSFORMS ############################################################################
##############################################################################################################
##############################################################################################################

# RandomSpatialFlip
# RandomRotate
# RandomInterpolate
# RandomTemporalFlip
# RandomShear
# RandomWarp

class AN_NTU_AE_Transforms_1:

    def __init__(self):

        self.transform = transform_lib.Compose([
            RandomSpatialFlip(prob=0.3),
            RandomRotate(prob=0.3),
            RandomShear(prob=0.3),
            RandomTemporalFlip(prob=0.3),
            RandomInterpolate(prob=0.3, min_len=11, max_len=50),
            RandomWarp(prob=0.3),
            Center(),
            ToTensor(),
        ])
        self.gt_transform = transform_lib.Compose([
            ToTensor(),
        ])

    def __call__(self, inp):
        return self.transform(inp)

class AN_NTU_AE_Transforms_1_gt(AN_NTU_AE_Transforms_1):

    def __call__(self, inp):
        q = self.transform(inp)
        gt = self.gt_transform(inp)
        return q, gt

class AN_NTU_AE_Transforms_2(AN_NTU_AE_Transforms_1):

    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k

class AN_NTU_AE_Transforms_2_gt(AN_NTU_AE_Transforms_1):

    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        gt = self.gt_transform(inp)
        return q, k, gt

class AN_NTU_All(AN_NTU_AE_Transforms_1):
    def __init__(self):
        self.transform = transform_lib.Compose([
            RandomSpatialFlip(prob=0.3),
            RandomRotate(prob=0.3),
            RandomShear(prob=0.3),
            RandomTemporalFlip(prob=0.3),
            RandomInterpolate(prob=0.3, min_len=31, probs=probs/probs.sum()),
            RandomWarp(prob=0.3),
            Center(),
            ToTensor(),
        ])

class AN_NTU_SpatialFlip(AN_NTU_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            SpatialFlip(),
            Center(),
            ToTensor(),
        ])

class AN_NTU_SpatialFlip_AX0(AN_NTU_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            SpatialFlip(p=1,axis=[0]),
            Center(),
            ToTensor(),
        ])

class AN_NTU_SpatialFlip_AX1(AN_NTU_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            SpatialFlip(p=1,axis=[1]),
            Center(),
            ToTensor(),
        ])

class AN_NTU_Rotate(AN_NTU_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            Rotate(),
            Center(),
            ToTensor(),
        ])

class AN_NTU_Shear(AN_NTU_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            Shear(),
            Center(),
            ToTensor(),
        ])

class AN_NTU_TemporalFlip(AN_NTU_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            TemporalFlip(),
            Center(),
            ToTensor(),
        ])

class AN_NTU_TemporalFlip_P1(AN_NTU_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            TemporalFlip(p=1),
            Center(),
            ToTensor(),
        ])

class AN_NTU_Interpolate(AN_NTU_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            Interpolate(min_len=31, probs=probs/probs.sum()),
            Center(),
            ToTensor(),
        ])

class AN_NTU_Warp(AN_NTU_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            Warp(),
            Center(),
            ToTensor(),
        ])

class AN_NTU_base(AN_NTU_AE_Transforms_1):
    def __init__(self):
        super().__init__()
        self.transform = transform_lib.Compose([
            Center(),
            ToTensor(),
        ])