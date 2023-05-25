import os
import sys

import cv2
import numpy as np
import torch
from smplx.lbs import batch_rodrigues

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from scipy.spatial.distance import cdist

from PointRend.point_rend import add_pointrend_config
from DensePose.densepose import add_densepose_config

import config

from predict.predict_joints2D import predict_joints2D
from predict.predict_silhouette_pointrend import predict_silhouette_pointrend
from predict.predict_densepose import predict_densepose

from models.smpl_official import SMPL
from renderers.weak_perspective_pyrender_renderer import Renderer

from utils.image_utils import pad_to_square, crop_and_resize_silhouette_joints
from utils.cam_utils import orthographic_project_torch
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_multiclass_to_binary_labels, \
    convert_2Djoints_to_gaussian_heatmaps
from utils.rigid_transform_utils import rot6d_to_rotmat

import matplotlib

# matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean

def limits(point, eps):
    # Ограничения по x, y и z
    x_min = point[:, 0] + eps
    x_max = 1.5
    y_min = point[:, 1] - eps
    y_max = point[:, 1] + eps
    z_min = point[:, 2] - eps
    z_max = point[:, 2] + eps
    return x_min, x_max, y_min, y_max, z_min, z_max


def create_mask(x_min, x_max, y_min, y_max, z_min, z_max, test_smpl_vertices):
    # Создание маски
    mask = np.where(
        (test_smpl_vertices[:, 0] >= x_min) & (test_smpl_vertices[:, 0] <= x_max) &
        (test_smpl_vertices[:, 1] >= y_min) & (test_smpl_vertices[:, 1] <= y_max) &
        (test_smpl_vertices[:, 2] >= z_min) & (test_smpl_vertices[:, 2] <= z_max)
    )
    return mask
def get_distance(first_point, second_point):
    distance = euclidean(first_point, second_point)
    return distance
def get_distances(point, filtered_points):
    distances = cdist(point, filtered_points)
    return distances

def get_nearest_index(distances):
    near = np.argmin(distances)
    return near

def setup_detectron2_predictors(silhouettes_from='densepose'):
    # Keypoint-RCNN
    kprcnn_config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    kprcnn_cfg = get_cfg()
    kprcnn_cfg.merge_from_file(model_zoo.get_config_file(kprcnn_config_file))
    kprcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    kprcnn_cfg.MODEL.DEVICE = "cpu"  # DEN DEBUG
    kprcnn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(kprcnn_config_file)
    kprcnn_cfg.freeze()
    joints2D_predictor = DefaultPredictor(kprcnn_cfg)

    if silhouettes_from == 'pointrend':
        # PointRend-RCNN-R50-FPN
        pointrend_config_file = "PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
        pointrend_cfg = get_cfg()
        add_pointrend_config(pointrend_cfg)
        pointrend_cfg.merge_from_file(pointrend_config_file)
        pointrend_cfg.MODEL.WEIGHTS = "checkpoints/pointrend_rcnn_R_50_fpn.pkl"
        pointrend_cfg.MODEL.DEVICE = "cpu"  # DEN DEBUG
        pointrend_cfg.freeze()
        silhouette_predictor = DefaultPredictor(pointrend_cfg)
    elif silhouettes_from == 'densepose':
        # DensePose-RCNN-R101-FPN
        densepose_config_file = "DensePose/configs/densepose_rcnn_R_101_FPN_s1x.yaml"
        densepose_cfg = get_cfg()
        densepose_cfg.MODEL.DEVICE = "cpu"  # DEN DEBUG
        add_densepose_config(densepose_cfg)
        densepose_cfg.merge_from_file(densepose_config_file)
        densepose_cfg.MODEL.WEIGHTS = "checkpoints/densepose_rcnn_R_101_fpn_s1x.pkl"
        densepose_cfg.freeze()
        silhouette_predictor = DefaultPredictor(densepose_cfg)

    return joints2D_predictor, silhouette_predictor


def create_proxy_representation(silhouette,
                                joints2D,
                                out_wh):
    heatmaps = convert_2Djoints_to_gaussian_heatmaps(joints2D.astype(np.int16),
                                                     out_wh)
    proxy_rep = np.concatenate([silhouette[:, :, None], heatmaps], axis=-1)
    proxy_rep = np.transpose(proxy_rep, [2, 0, 1])  # (C, out_wh, out_WH)

    return proxy_rep


def predict_3D(input,
               regressor,
               device,
               silhouettes_from='densepose',
               proxy_rep_input_wh=512,
               save_proxy_vis=True,
               render_vis=True):
    # Set-up proxy representation predictors.
    joints2D_predictor, silhouette_predictor = setup_detectron2_predictors(silhouettes_from=silhouettes_from)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    if render_vis:
        # Set-up renderer for visualisation.
        wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    if os.path.isdir(input):
        image_fnames = [f for f in sorted(os.listdir(input)) if f.endswith('.png') or
                        f.endswith('.jpg')]
        for fname in image_fnames:
            print("Predicting on:", fname)
            image = cv2.imread(os.path.join(input, fname))
            # Pre-process for 2D detectors
            image = pad_to_square(image)
            image = cv2.resize(image, (proxy_rep_input_wh, proxy_rep_input_wh),
                               interpolation=cv2.INTER_LINEAR)
            # Predict 2D
            joints2D, joints2D_vis = predict_joints2D(image, joints2D_predictor)
            if silhouettes_from == 'pointrend':
                silhouette, silhouette_vis = predict_silhouette_pointrend(image,
                                                                          silhouette_predictor)
            elif silhouettes_from == 'densepose':
                silhouette, silhouette_vis = predict_densepose(image, silhouette_predictor)
                silhouette = convert_multiclass_to_binary_labels(silhouette)
            # Crop around silhouette
            silhouette, joints2D, image = crop_and_resize_silhouette_joints(silhouette,
                                                                            joints2D,
                                                                            out_wh=config.REGRESSOR_IMG_WH,
                                                                            image=image,
                                                                            image_out_wh=proxy_rep_input_wh,
                                                                            bbox_scale_factor=1.2)

            print(type(joints2D), joints2D)

            left_sh = (joints2D[5, 0], joints2D[5, 1])
            right_sh = (joints2D[6, 0], joints2D[6, 1])

            def max_consecutive_ones_in_row(arr, row_index):
                # Получаем указанную строку
                row = arr[row_index, :]

                # Находим разности между соседними элементами в строке
                diffs = np.diff(row)

                # Находим индексы, где происходит изменение значения (из 1 в 0 или из 0 в 1)
                change_indices = np.where(diffs != 0)

                # Вычисляем длины последовательностей единиц
                consecutive_ones_lengths = np.diff(change_indices)

                # Находим максимальную длину последовательности единиц
                max_consecutive_ones = np.max(consecutive_ones_lengths)

                return max_consecutive_ones

            # Пример использования

            row_index = int((joints2D[11, 1] + joints2D[12, 1]) / 2)
            max_ones = max_consecutive_ones_in_row(silhouette, row_index)
            print(silhouette[row_index, :])
            print(max_ones)

            # Create proxy representation
            proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                    out_wh=config.REGRESSOR_IMG_WH)

            proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension

            proxy_rep = torch.from_numpy(proxy_rep).float().to(device)


            # Predict 3D
            regressor.eval()
            with torch.no_grad():

                print(f"proxy_rep [0,0]: {proxy_rep[0, 0]}, shape: {proxy_rep[0, 0].shape}, max_el: {torch.max(proxy_rep[0, 0])}")
                pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)
                # Convert pred pose to rotation matrices
                if pred_pose.shape[-1] == 24 * 3:
                    pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                    pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                elif pred_pose.shape[-1] == 24 * 6:
                    print("pred_pose", pred_pose)
                    pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)
                    print(pred_pose_rotmats.shape)
                    print("pred_pose_rotmats[:, 1:]", pred_pose_rotmats[:, 1:])
                    print("shape", pred_pose_rotmats[:, 1:].shape)

                pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                        global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                        betas=pred_shape,
                                        pose2rot=False)

                # sys.exit()

                # f = open("/home/nata/pythonProj/STRAPS/demofile.txt", "w")
                # f.write(pred_smpl_output)
                # f.close()
                # print(pred_shape)
                # a_list = pred_shape.tolist()
                # print(a_list[0])
                with open("/home/nata/pythonProj/STRAPS/smpl_result.txt", "a") as file:
                    a_list = pred_shape[0].tolist()
                    pred_str = ",".join(map(str, a_list))
                    file.write(fname + "," + pred_str)
                    file.write("\n")
                pred_vertices = pred_smpl_output.vertices
                pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
                pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                              proxy_rep_input_wh)

                #kamil_test________________________________
                test_body_pose = torch.zeros([1, 69], dtype=torch.float32)

                test_body_pose[0][50] = 0.7
                test_body_pose[0][47] = -0.7

                test_reposed_smpl_output = smpl(betas=pred_shape, body_pose=test_body_pose)
                test_smpl_output = test_reposed_smpl_output.vertices


                test_smpl_vertices = test_smpl_output.cpu().detach().numpy()[0]


                max_x = np.amax(test_smpl_vertices[:,0])  # Максимальные значения по оси x
                max_y = np.amax(test_smpl_vertices[:,1])  # Максимальные значения по оси x
                max_z = np.amax(test_smpl_vertices[:,2])  # Максимальные значения по оси x

                min_x = np.amin(test_smpl_vertices[:, 0])  # Максимальные значения по оси x
                min_y = np.amin(test_smpl_vertices[:, 1])  # Максимальные значения по оси x
                min_z = np.amin(test_smpl_vertices[:, 2])  # Максимальные значения по оси x


                # for i in range(69):
                #     test_body_pose = torch.zeros([1, 69], dtype=torch.float32)
                #     test_body_pose[0][i] = 0.5
                #     test_reposed_smpl_output = smpl(betas=pred_shape, body_pose=test_body_pose)
                #
                #     test_smpl_output = test_reposed_smpl_output.vertices
                #
                #     test_smpl_vertices = test_smpl_output.cpu().detach().numpy()[0]
                #
                #     rend_reposed_img = wp_renderer.render(verts=test_smpl_vertices,
                #                                           cam=np.array([0.8, 0., -0.2]),
                #                                           angle=180,
                #                                           axis=[1, 0, 0])
                #     cv2.imwrite(os.path.join(input, 'rend_vis/kamil_test', f'point_{i}_reposed_{fname}'), rend_reposed_img)
                #
                # sys.exit()
                #kamil test end________________________________________

                pred_reposed_smpl_output = smpl(betas=pred_shape)
                #joints = pred_reposed_smpl_output.joints[:,:24,:]
                joints = test_reposed_smpl_output.joints.numpy()[:,:24,:]
                shoulder_left = joints[:,16,:]
                hip_left = joints[:,1,:]
                print((shoulder_left.shape))
                shoulder_right = joints[:,17,:]
                eps = 0.02

                x_min, x_max, y_min, y_max, z_min, z_max = limits(shoulder_left, eps)


                # Создание маски
                mask = create_mask(x_min, x_max, y_min, y_max, z_min, z_max, test_smpl_vertices)

                # Извлечение точек, удовлетворяющих ограничениям
                filtered_points = test_smpl_vertices[mask]

                print("joints", joints)
                pred_reposed_vertices = pred_reposed_smpl_output.vertices

                # Создание трехмерного графика
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')

                x = test_smpl_vertices[:, 0]
                y = test_smpl_vertices[:, 1]
                z = test_smpl_vertices[:, 2]

                # ax2 = fig.add_subplot(projection='3d')

                x2 = joints[:,:, 0]
                y2 = joints[:,:, 1]
                z2 = joints[:,:, 2]

                x3, y3, z3 = filtered_points.T
                print(x3.shape)

                # Построение точек
                # ax.scatter(x, y, z)


                # Настройка осей
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.axes.set_xlim3d(left=-1.1, right=1.1)
                ax.axes.set_ylim3d(bottom=-1.1, top=1.1)
                ax.axes.set_zlim3d(bottom=-1.1, top=1.1)


                # Вычисление евклидовых расстояний между заданной точкой и filtered_points
                shoulder_distances = get_distances(shoulder_left, filtered_points)

                # Нахождение индекса ближайшей точки
                shoulder_nearest_index = get_nearest_index(shoulder_distances)

                # Получение ближайшей точки
                shoulder_nearest_point = filtered_points[shoulder_nearest_index]
                print(shoulder_nearest_point)

                shoulder_distance = get_distance(shoulder_nearest_point, [0, shoulder_nearest_point[1],shoulder_nearest_point[2]])
                print (shoulder_distance*2)



                ax.scatter(x2, y2, z2, c='r', marker='o')
                # ax.scatter(x3, y3, z3, c='k', marker='o')
                ax.scatter(shoulder_nearest_point[0], shoulder_nearest_point[1],shoulder_nearest_point[2], c='k', marker='o')
                ax.scatter(0, shoulder_nearest_point[1],shoulder_nearest_point[2], c='b', marker='o')

                plt.show()



                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(joints[:,:, 0], joints[:,:, 1], joints[:,:, 2], c='r', marker='o')
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
                # ax.set_title('3D Joints Visualization')
                #
                #
                #
                # ax.set_box_aspect([1, 1, 1])  # Set equal aspect ratio for all three axes
                # ax.set_xlim([-1, 1])  # Set x-axis limits
                # ax.set_ylim([-1, 1])  # Set y-axis limits
                # ax.set_zlim([-1, 1])  # Set z-axis limits
                #
                # for i in range(joints.shape[1]):
                #     ax.text(joints[:, i, 0], joints[:, i, 1], joints[:, i, 2], str(i), color='black')
                #
                # plt.show()

                # ax.view_init(elev=90, azim=0)  # Top view
                # plt.savefig('top_view.png')
                #
                # ax.view_init(elev=0, azim=0)  # Front view
                # plt.savefig('front_view.png')
                #
                # ax.view_init(elev=0, azim=90)  # Side view
                # plt.savefig('side_view.png')


                pred_reposed_smpl_output.body_pose.data[0][0] = 1
                print("----")
            # Numpy-fying
            pred_vertices = pred_vertices.cpu().detach().numpy()[0]
            pred_vertices2d = pred_vertices2d.cpu().detach().numpy()[0]
            print(pred_reposed_vertices.numel())
            pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()[0]
            print(pred_reposed_vertices.shape)
            pred_cam_wp = pred_cam_wp.cpu().detach().numpy()[0]

            if not os.path.isdir(os.path.join(input, 'verts_vis')):
                os.makedirs(os.path.join(input, 'verts_vis'))
            plt.figure()
            plt.imshow(image[:, :, ::-1])
            plt.scatter(pred_vertices2d[:, 0], pred_vertices2d[:, 1], s=0.3)
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(os.path.join(input, 'verts_vis', 'verts_' + fname))

            if render_vis:
                rend_img = wp_renderer.render(verts=pred_vertices, cam=pred_cam_wp, img=image)
                # rend_reposed_img = wp_renderer.render(verts=pred_reposed_vertices,
                #                                       cam=np.array([0.8, 0., -0.2]),
                #                                       angle=180,
                #                                       axis=[1, 0, 0])
                rend_reposed_img = wp_renderer.render(verts=test_smpl_vertices,
                                                          cam=np.array([0.8, 0., -0.2]),
                                                          angle=180,
                                                          axis=[1, 0, 0])
                if not os.path.isdir(os.path.join(input, 'rend_vis')):
                    os.makedirs(os.path.join(input, 'rend_vis'))
                cv2.imwrite(os.path.join(input, 'rend_vis', 'rend_' + fname), rend_img)
                cv2.imwrite(os.path.join(input, 'rend_vis', 'reposed_' + fname), rend_reposed_img)
            if save_proxy_vis:
                if not os.path.isdir(os.path.join(input, 'proxy_vis')):
                    os.makedirs(os.path.join(input, 'proxy_vis'))
                cv2.imwrite(os.path.join(input, 'proxy_vis', 'silhouette_' + fname), silhouette_vis)
                cv2.imwrite(os.path.join(input, 'proxy_vis', 'joints2D_' + fname), joints2D_vis)

            sys.exit()