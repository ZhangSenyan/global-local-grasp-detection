
import numpy as np
import math
import cv2

def get_pointcloud(color_img, depth_img, camera_intrinsics,cam_pose):

    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h))

    cam_pts_x = np.multiply(pix_x - camera_intrinsics[0][2], depth_img / camera_intrinsics[0][0])  # 对应元素相乘
    cam_pts_y = np.multiply(pix_y - camera_intrinsics[1][2], depth_img / camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()

    cam_pts_x.shape = (im_h * im_w, 1)
    cam_pts_y.shape = (im_h * im_w, 1)
    cam_pts_z.shape = (im_h * im_w, 1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:, :, 0]
    rgb_pts_g = color_img[:, :, 1]
    rgb_pts_b = color_img[:, :, 2]
    rgb_pts_r.shape = (im_h * im_w, 1)
    rgb_pts_g.shape = (im_h * im_w, 1)
    rgb_pts_b.shape = (im_h * im_w, 1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    surface_pts = np.transpose(np.dot(cam_pose[0:3, 0:3], np.transpose(cam_pts)) + np.tile(cam_pose[0:3, 3:], (1, cam_pts.shape[0])))

    return surface_pts,rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):

    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:,2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]

    depth_heightmap[depth_heightmap < 0] = 0


    return color_heightmap, depth_heightmap



def Creat_posemat(orientation,position):
    C=math.cos; S=math.sin;
    alpha=orientation[0];beta=orientation[1];gamma=orientation[2]
    P=position
    RX=np.array([[1,                 0,               0],
                 [0,   math.cos(alpha),-math.sin(alpha)],
                 [0,   math.sin(alpha), math.cos(alpha)]])
    RY = np.array([[ math.cos(beta), 0,          math.sin(beta)],
                   [              0, 1,                       0],
                   [-math.sin(beta), 0,         math.cos(beta)]])
    RZ=np.array([[math.cos(gamma),-math.sin(gamma),0],
                 [math.sin(gamma), math.cos(gamma),0],
                 [              0,               0,1]])
    mat = np.eye(4)
    mat[:3, :3]=np.dot(np.dot(RX,RY),RZ)
    mat[:3,3]=np.array(position)
    return mat


def GraspBaseFeature(heightmap):

    maxscore=0
    maxarg=[120,120,0]
    maxrectSlide=recttailor(heightmap, 120, 120, 0)
    for r in range(0,heightmap.shape[0],20):
        for c in range(0,heightmap.shape[1],20):
            print("r:",r," c:",c)
            for angle in range(0,180,10):
                rectSlide=recttailor(heightmap, r, c, angle)
                score=getGraspRectScore(rectSlide)
                if score>maxscore:
                    maxscore=score
                    maxarg = [r, c, angle]
                    maxrectSlide=rectSlide
                    print(score)
    return maxarg,maxrectSlide


import random
def GraspBaseFeature2(heightmap,color_image,threshold=1000):

    print("GraspBaseFeature2 start")
    maxscore=0
    maxarg=[120,120,0]
    maxrectSlide=recttailor(heightmap, 120, 120, 0)
    for i in range(1000):
        r=random.randint(0,239)
        c = random.randint(0, 239)
        angle=random.randint(0, 180)
        rectSlide=recttailor(heightmap, r, c, angle)
        score=getGraspRectScore(rectSlide)
        if score>maxscore:
            maxscore=score
            maxarg = [r, c, angle]
            maxrectSlide=rectSlide
            #print(score)
        if maxscore>threshold:
            break
    colorrectslide=recttailor(color_image, maxarg[0], maxarg[1],maxarg[2],True)
    print("GraspBaseFeature2 end")
    return maxarg,maxrectSlide,colorrectslide


def recttailor(heightmap,r,c,rotangle,color=False):
    rotrad=rotangle/180*math.pi
    if color:
        rectslide = np.zeros((36,72,3), np.uint8)
    else:
        rectslide = np.zeros((36, 72), np.float)

    for x in range(int(rectslide.shape[0])):
        for y in range(int(rectslide.shape[1])):

            l2 = pow((x - rectslide.shape[0] / 2) * (x - rectslide.shape[0] / 2) + (y - rectslide.shape[1] / 2) * (y - rectslide.shape[1] / 2),
                     0.5)

            if y!=int(rectslide.shape[1] / 2):
                if float(y) - rectslide.shape[1] / 2. <= 0:
                    baserad = math.atan((float(x) - rectslide.shape[0] / 2.) / (float(y) - rectslide.shape[1] / 2.))
                else:
                    baserad = math.atan((float(x) - rectslide.shape[0] / 2.) / (float(y) - rectslide.shape[1] / 2.)) + math.pi

            else:
                if x<=int(rectslide.shape[1] / 2):
                    baserad=math.pi/2
                else:
                    baserad=-math.pi/2

            rad = rotrad + baserad

            hm_x = round(r - math.sin(rad) * l2)
            hm_y = round(c - math.cos(rad) * l2)

            if hm_x >= 0 and hm_x < heightmap.shape[0] and hm_y >= 0 and hm_y < heightmap.shape[1]:
                rectslide[x][y] = heightmap[hm_x][hm_y]
    return rectslide


def getGraspRectScore(rectSlide):
    rectSlide=rectSlide[:, :] / np.max(rectSlide) * 1  #归一化
    leftrect=rectSlide[:, :20]
    middlerect=rectSlide[:, 36-10:36+10]
    rightrect=rectSlide[:, 72-20:]
    score=np.sum(middlerect)-np.sum(leftrect)+np.sum(middlerect)-np.sum(rightrect)
    return score



def load_patches(data_path_base):
    y=[]
    x=[]
    with open(data_path_base+'lable.txt') as f:
        for line in f.readlines():
            y.append([int(line.split()[1])])
            img=cv2.imread(data_path_base+"depth_rect_image/depth_rect_img_"+line.split()[0][10:],cv2.IMREAD_GRAYSCALE)
            x.append((img[:,:]*0.12/255.).reshape(36,72,1))

    X=np.array(x)
    Y=np.array(y)
    X_train=X[0:int(X.shape[0]*4/5)]
    Y_train=Y[0:int(X.shape[0]*4/5)]
    X_test =X[int(X.shape[0] * 4 / 5):]
    Y_test = Y[int(X.shape[0] * 4 / 5):]
    return (X_train,Y_train),(X_test,Y_test)





def GetRoatePatch(location,angles,depth_heightmap):

    inputImage = np.zeros([240 + 72, 240 + 72])
    inputImage[37:37 + 240, 37:37 + 240] = depth_heightmap

    r = location[0]+36
    c = location[1]+36
    pitch = inputImage[r - 36:r + 36, c - 36:c + 36]
    #print(time.time())

    rectSlides = []
    for angle in angles:
        img_rote = ndimage.rotate(pitch, angle)
        #print(img_rote.shape)
        # print(time.time())
        center = int(img_rote.shape[0] / 2)
        rect = img_rote[center - 18:center + 18, center - 36:center + 36]
        #print(rect.shape)
        rectSlides.append(rect[:, :] - np.mean(rect))

    rectSlides=np.array(rectSlides)

    return rectSlides

