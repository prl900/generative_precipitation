import numpy as np
from keras.utils import Sequence
import xarray as xr

class DataGenerator(Sequence):
    def __init__(self, partition='train', batch_size=32, n_channels=3, t_stride=10, shuffle=True, seed=1):

        self.partition = partition
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.t_stride = t_stride
        self.shuffle = shuffle
        self.era5 = None
        self.prec = None
        #self.on_epoch_end()

        # Load ERA5 geopotential levels
        era5_ds1 = xr.open_dataset("./datasets/GEOP1000_GAN_2017.nc")
        era5_ds2 = xr.open_dataset("./datasets/GEOP800_GAN_2017.nc")
        era5_ds3 = xr.open_dataset("./datasets/GEOP500_GAN_2017.nc")
        era5_times = era5_ds1.time[:].data

        # Load ERA5 total precipitation
        prec_ds = xr.open_dataset("./datasets/TP_GAN_2017.nc")
        prec_times = prec_ds.time[:].data

        # Find common dates and shuffle
        times = np.intersect1d(era5_times, prec_times)
        np.random.seed(seed)
        np.random.shuffle(times)

        # Create geopotential normalised stack
        z500 = era5_ds3.Geopotential.sel(time=times[::self.t_stride])[:].data
        z500 = (z500 - z500.min()) / (z500.max() - z500.min())
        z800 = era5_ds2.Geopotential.sel(time=times[::self.t_stride])[:].data
        z800 = (z800 - z800.min()) / (z800.max() - z800.min())
        z1000 = era5_ds1.Geopotential.sel(time=times[::self.t_stride])[:].data
        z1000 = (z1000 - z1000.min()) / (z1000.max() - z1000.min())
        self.era5 = np.stack((z1000, z800, z500), axis=3)
        z1000, z800, z500 = None, None, None
        self.era5 = (self.era5 * 2) - 1

        # Create precipitation normalised stack
        tp = prec_ds.tp.sel(time=times[::self.t_stride])[:].data * 1000
        tp = np.clip(tp, 0, 30)
        tp1 = np.log(1+np.log(1+tp))
        tp1 = np.clip(tp1, 0, 1)
        tp2 = np.log(1+tp)/np.log(31)
        tp3 = tp / 30

        self.prec = np.stack((tp1, tp2, tp3), axis=3)
        tp, tp1, tp2, tp3 = None, None, None, None
        self.prec = (self.prec * 2) - 1

        if self.partition == 'train':
            n1 = int(tp.shape[0] * .7)
            self.era5 = self.era5[:n1,:,:,:]
            self.prec = self.prec[:n1,:,:,:]

        elif self.partition == 'test':
            n0 = int(tp.shape[0] * .7)
            n1 = int(tp.shape[0] * .9)
            self.era5 = self.era5[n0:n1,:,:,:]
            self.prec = self.prec[n0:n1,:,:,:]

        elif self.partition == 'val':
            n0 = int(tp.shape[0] * .9)
            self.era5 = self.era5[n0:,:,:]
            self.prec = self.prec[n0:,:,:]


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        idx = np.random.randint(self.prec.shape[0], size=self.batch_size)

        return self.era5[idx,:,:], self.prec[idx,:,:]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass
