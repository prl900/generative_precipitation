import numpy as np
import xarray as xr

class DataLoader():
    def __init__(self):

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
        np.random.shuffle(times)

        # Create geopotential normalised stack
        z500 = era5_ds3.Geopotential.sel(time=times[::10])[:].data
        z500 = (z500 - z500.min()) / (z500.max() - z500.min())
        z800 = era5_ds2.Geopotential.sel(time=times[::10])[:].data
        z800 = (z800 - z800.min()) / (z800.max() - z800.min())
        z1000 = era5_ds1.Geopotential.sel(time=times[::10])[:].data
        z1000 = (z1000 - z1000.min()) / (z1000.max() - z1000.min())
        z = np.stack((z1000, z800, z500), axis=3)
        z = (z * 2) - 1

        # Create precipitation normalised stack
        tp = prec_ds.tp.sel(time=times[::10])[:].data * 1000
        tp = np.clip(tp, 0, 30)
        tp1 = tp / 30
        tp2 = np.log(1+tp)/np.log(31)
        tp3 = np.log(1+np.log(1+tp))
        tp3 = np.clip(tp3, 0, 1)

        tp = np.stack((tp3, tp3, tp3), axis=3)
        tp = (tp * 2) - 1

        self.prec_train = tp[:600,:,:,:]
        self.era5_train = z[:600,:,:,:]

        self.prec_test = tp[600:750,:,:,:]
        self.era5_test = z[600:750,:,:,:]

        self.prec_val = tp[750:,:,:]
        self.era5_val = z[750:,:,:]


    def load_data(self, batch_size=1, is_testing=False):
        if is_testing:
            idx = np.random.choice(self.prec_test.shape[0], size=batch_size)
            return self.prec_test[idx,:,:,:], self.era5_test[idx,:,:,:]
        else:
            idx = np.random.choice(self.prec_train.shape[0], size=batch_size)
            return self.prec_train[idx,:,:,:], self.era5_train[idx,:,:,:]


    def load_batch(self, batch_size=1, is_testing=False):
        prec_data = None
        him_data = None

        if is_testing:
            prec_data = self.prec_test
            him_data = self.era5_test
        else:
            prec_data = self.prec_train
            him_data = self.era5_train

        self.n_batches = int(prec_data.shape[0] / batch_size)

        for i in range(self.n_batches-1):
            yield prec_data[i*batch_size:(i+1)*batch_size,:,:,:], him_data[i*batch_size:(i+1)*batch_size,:,:,:]

