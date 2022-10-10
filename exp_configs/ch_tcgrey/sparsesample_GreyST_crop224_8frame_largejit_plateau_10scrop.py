_exec_relative_('sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py')

sampling_mode = 'GreyST'
greyscale=True

def _dataloader_shape_to_model_input_shape(inputs):
    N, C, T, H, W = inputs.shape        # C = 1
    return inputs.view((N,3,T//3,H,W)).reshape((N,-1,H,W))

