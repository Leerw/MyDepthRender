#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

pcl::PointCloud<pcl::PointXYZ>::Ptr part_cloud, complete_cloud;

double sum_x{0}, sum_y{0}, sum_z{0};
float mean_x{0}, mean_y{0}, mean_z{0};
float max_x{0}, max_y{0}, max_z{0};
float radius = 0.5;
float scale_x{1}, scale_y{1}, scale_z{1};
const float eps = 1e-3;
const float zNear = std::sqrt(3) - radius - eps;
const float zFar = zNear + 1 + eps;

glm::vec3 camPos;
glm::vec3 centerPos(0.0, 0.0, 0.0);
std::string completeDir;
std::string partDir;
int currentView = 0;
std::vector<glm::vec3> camPosList;

bool mode = true;

pcl::PointCloud<pcl::PointXYZ>::Ptr readPCD(string filename, bool isComplete) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
    PCL_ERROR("Could not read file\n");
    return nullptr;
  }

  std::cout << "Loaded " << filename << " " << cloud->points.size() << " points"
            << std::endl;

  // translate, normalize and scale pcd
  if (isComplete) {
    for (size_t i = 0; i < cloud->points.size(); i++) {
      sum_x += cloud->points[i].x;
      sum_y += cloud->points[i].y;
      sum_z += cloud->points[i].z;
    }
    mean_x = sum_x / cloud->points.size();
    mean_y = sum_y / cloud->points.size();
    mean_z = sum_z / cloud->points.size();
  }

  for (size_t i = 0; i < cloud->points.size(); i++) {
    cloud->points[i].x -= mean_x;
    cloud->points[i].y -= mean_y;
    cloud->points[i].z -= mean_z;
    if (isComplete) {
      max_x = cloud->points[i].x > max_x ? cloud->points[i].x : max_x;
      max_y = cloud->points[i].y > max_y ? cloud->points[i].y : max_y;
      max_z = cloud->points[i].z > max_z ? cloud->points[i].z : max_z;
    }
  }

  if (isComplete) {
    scale_x = max_x / radius;
    scale_y = max_y / radius;
    scale_z = max_z / radius;
  }

  for (size_t i = 0; i < cloud->points.size(); i++) {
    cloud->points[i].x /= scale_x;
    cloud->points[i].y /= scale_y;
    cloud->points[i].z /= scale_z;
  }

  return cloud;
}

void display() {
  if (currentView >= camPosList.size() * 2) {
    exit(0);
  }

  // render partial point cloud
  if (currentView >= camPosList.size()) {
    mode = false;
  }

  int w = glutGet(GLUT_WINDOW_WIDTH);
  int h = glutGet(GLUT_WINDOW_HEIGHT);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  double ar = w / static_cast<double>(h);
  gluPerspective(43, ar, zNear, zFar); // simulate kinect
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glm::vec3 eye = camPosList[currentView % camPosList.size()];
  gluLookAt(eye[0], eye[1], eye[2], centerPos[0], centerPos[1], centerPos[2], 0,
            1, 0);
  static float angle = 0;
  glColor3ub(255, 0, 0);

  // render point cloud
  glBegin(GL_POINTS);
  if (true == mode) {
    for (size_t i = 0; i < complete_cloud->points.size(); i++) {
      glVertex3f(complete_cloud->points[i].x, complete_cloud->points[i].y,
                 complete_cloud->points[i].z);
    }
  } else {
    for (size_t i = 0; i < part_cloud->points.size(); i++) {
      glVertex3f(part_cloud->points[i].x, part_cloud->points[i].y,
                 part_cloud->points[i].z);
    }
  }
  glEnd();
  glPopMatrix();

  vector<GLfloat> depth(w * h, 0);
  glReadPixels(0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT,
               &depth[0]); // read depth buffer
  cv::Mat img(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH),
              CV_32FC3); // output depth image
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      depth[i * img.cols + j] =
          (2.0 * zNear * zFar) /
          (zFar + zNear -
           (2.0f * depth[i * img.cols + j] - 1) * (zFar - zNear)); // [zNear, zFar]
      depth[i * img.cols + j] =
          (depth[i * img.cols + j] - zNear) / (zFar - zNear); // [0, 1]
      img.at<float>(i, j) = depth[i * img.cols + j] * 255;
    }
  }
  cv::Mat flipped(img);
  cv::flip(img, flipped, 0);

  cv::Mat imgRGB(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH),
                 CV_32FC3);
  // output depth image
  for (int i = 0; i < imgRGB.rows; i++) {
    for (int j = 0; j < imgRGB.cols; j++) {
      imgRGB.at<cv::Vec3f>(i, j) = cv::Vec3f(
          img.at<float>(i, j), img.at<float>(i, j), img.at<float>(i, j));
    }
  }

  std::string currentName;
  if (true == mode) {
    currentName = completeDir + "//complete" + +"_cam_" +
                  to_string(currentView % camPosList.size()) + ".png";
  } else {
    currentName = partDir + "//part" + +"_cam_" +
                  to_string(currentView % camPosList.size()) + ".png";
  }
  cv::imwrite(currentName, imgRGB);

  currentView++;
  glutSwapBuffers();
}

void timer(int value) {
  glutPostRedisplay();
  glutTimerFunc(8, timer, 0);
}

int main(int argc, char **argv) {
  // parse args
  std::string completePCD = argv[1];
  std::string partPCD = argv[2];
  std::string cameraFile = argv[3];
  completeDir = argv[4];
  partDir = argv[5];
  int width = stoi(argv[6]);
  int height = stoi(argv[7]);

  // read point cloud
  complete_cloud = readPCD(completePCD, true);
  part_cloud = readPCD(partPCD, false);

  // read camera extrinsic parameters
  float x, y, z;
  ifstream fin(cameraFile);
  while (fin >> x >> y >> z) {
    glm::vec3 c(x, y, z);
    camPosList.push_back(c);
  }

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(width, height);
  glutCreateWindow("GLUT");
  glewInit();
  glutDisplayFunc(display);
  glutTimerFunc(0, timer, 0);
  glEnable(GL_DEPTH_TEST);
  glutMainLoop();
  return 0;
}