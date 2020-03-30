#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include <cmath>
#include <iostream>
#include <vector>


using namespace std;
using namespace cv;
using namespace glm;

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
vec3 camPos;
vec3 centerPos(0.0, 0.0, 0.0);
string imageName;
string imagePath;
int currentView = 0;
float radius = 0.2;
vector<vec3> camPosList;

pcl::PointCloud<pcl::PointXYZ>::Ptr readPCD(string filename) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
    PCL_ERROR("Could not read file\n");
    return nullptr;
  }

  std::cout << "Loaded " << filename << " " << cloud->points.size() << " points"
            << std::endl;

  // translate, normalize and scale pcd
  double sum_x{0}, sum_y{0}, sum_z{0};
  float mean_x{0}, mean_y{0}, mean_z{0};
  for (size_t i = 0; i < cloud->points.size(); i++) {
    sum_x += cloud->points[i].x;
    sum_y += cloud->points[i].y;
    sum_z += cloud->points[i].z;
  }
  mean_x = sum_x / cloud->points.size();
  mean_y = sum_y / cloud->points.size();
  mean_z = sum_z / cloud->points.size();

  float max_x{0}, max_y{0}, max_z{0};

  for (size_t i = 0; i < cloud->points.size(); i++) {
    cloud->points[i].x -= mean_x;
    cloud->points[i].y -= mean_y;
    cloud->points[i].z -= mean_z;
    max_x = cloud->points[i].x > max_x ? cloud->points[i].x : max_x;
    max_y = cloud->points[i].y > max_y ? cloud->points[i].y : max_y;
    max_z = cloud->points[i].z > max_z ? cloud->points[i].z : max_z;
  }

  float scale_x{max_x / radius}, scale_y{max_y / radius},
      scale_z{max_z / radius};

  for (size_t i = 0; i < cloud->points.size(); i++) {
    cloud->points[i].x /= scale_x;
    cloud->points[i].y /= scale_y;
    cloud->points[i].z /= scale_z;
  }

  return cloud;
}

void display() {
  if (currentView >= camPosList.size()) {
    exit(0);
  }

  int w = glutGet(GLUT_WINDOW_WIDTH);
  int h = glutGet(GLUT_WINDOW_HEIGHT);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  double ar = w / static_cast<double>(h);
  const float zNear = 1;
  const float zFar = 2;
  gluPerspective(43, ar, zNear, zFar); // simulate kinect
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  vec3 eye = camPosList[currentView];
  gluLookAt(eye[0], eye[1], eye[2], centerPos[0], centerPos[1], centerPos[2], 0,
            1, 0);
  static float angle = 0;
  glColor3ub(255, 0, 0);

  // render point cloud
  glBegin(GL_POINTS);
  for (size_t i = 0; i < cloud->points.size(); i++) {
    glVertex3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
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
           (2.0f * depth[i * img.cols + j] - 1) * (zFar - zNear));
      depth[i * img.cols + j] =
          (depth[i * img.cols + j] - zNear) / (zFar - zNear);
      img.at<float>(i, j) = (1.0f - depth[i * img.cols + j]) * 255;
      // img.at<float>(i, j) = (1.0f - depth[i*img.cols + j]);
    }
  }
  cv::Mat flipped(img);
  cv::flip(img, flipped, 0);

  cv::Mat imgRGB(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH),
                 CV_32FC3); // output depth image
  for (int i = 0; i < imgRGB.rows; i++) {
    for (int j = 0; j < imgRGB.cols; j++) {
      imgRGB.at<cv::Vec3f>(i, j) =
          cv::Vec3f(img.at<float>(i, j), img.at<float>(i, j),
                    img.at<float>(i, j)); // flip image
    }
  }

  string currentName =
      imagePath + "//" + imageName + "_Cam_" + to_string(currentView) + ".png";
  cv::imwrite(currentName, imgRGB);

  currentView++;
  glutSwapBuffers();
}

void timer(int value) {
  glutPostRedisplay();
  glutTimerFunc(8, timer, 0);
}

int main(int argc, char **argv) {
  std::cout << "Start" << std::endl;

  cloud = readPCD(argv[1]);

  float x, y, z;
  ifstream fin(argv[2]);
  while (fin >> x >> y >> z) {
    vec3 c(x, y, z);
    camPosList.push_back(c);
  }
  imageName = string(argv[3]);
  imagePath = string(argv[4]);

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(256, 256);
  glutCreateWindow("GLUT");
  glewInit();
  glutDisplayFunc(display);
  glutTimerFunc(0, timer, 0);
  glEnable(GL_DEPTH_TEST);
  glutMainLoop();
  return 0;
}