import nibabel as nib
from lib.losses3D.basic import *


def visualize_3D_no_overlap(args, filename, full_volume, seg_map, model, affine, is_details=True):
    classes = args.classes
    num, modalities, height, width, slices = full_volume.shape
    predictions = model.inference(full_volume)
    full_vol_predictions = predictions.view(classes, width, height, slices)
    _, indices = full_vol_predictions.max(dim=0)
    full_vol_predictions_binary = indices.float()
    save_path = args.save
    if args.run_mode != 'test_without_label':
        seg_map_squeeze = seg_map.squeeze(0)
        dice = compute_per_channel_dice_eval(full_vol_predictions_binary, seg_map_squeeze)
        if is_details:
            print('Subject', filename, 'prediction Dice score: ', dice)
        summary = 'Subject '+filename+' prediction Dice score: '+np.array2string(dice)
        dice = np.around(dice, 4)
        if args.save_prediction:
            save_3d_vol(full_vol_predictions_binary.numpy(), affine, save_path + filename + '_' + str(dice))
        return summary, dice
    else:
        save_3d_vol(full_vol_predictions_binary.numpy(), affine, save_path + filename)


def get_prediction(output,classes, width, height, slices):
    full_vol_prediction = output.view(classes, width, height, slices)
    _, indices = full_vol_prediction.max(dim=0)
    full_vol_predictions_binary = indices.float()
    return full_vol_predictions_binary


def visualize_3D_no_overlap_fftn(args, filename, full_volume_low_list, full_volume_high, seg_map, model, affine, num_classes=2, is_details=True):
    classes = args.classes
    save_path = args.save
    num, modalities, height, width, slices = full_volume_high.shape
    prediction_list = model.inference(full_volume_low_list, full_volume_high)
    prediction = prediction_list[:,1]
    full_vol_predictions_binary = get_prediction(prediction,classes, width, height, slices)
    if args.run_mode != 'test_without_label':
        dice = compute_per_channel_dice_eval(full_vol_predictions_binary, seg_map)
        if is_details:
            print('Subject', filename, 'prediction Dice score: ', dice)
        summary = 'Subject ' + filename + ' prediction Dice score: ' + np.array2string(dice)
        dice = np.around(dice, 4)
        if args.save_prediction:
            save_3d_vol(full_vol_predictions_binary.numpy(), affine, save_path + filename + '_' + str(dice))
        return summary, dice
    else:
        save_3d_vol(full_vol_predictions_binary.numpy(), affine, save_path + filename)


def save_3d_vol(predictions, affine, save_path):
    pred_nifti_img = nib.Nifti1Image(predictions, affine)
    nib.save(pred_nifti_img, save_path + '.nii.gz')
