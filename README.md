# myrender

## prerequisites

- OpenGL
- GLEW
- GLUT
- GLM
- OpenCV
- PCL

## how to use

```bash
mkdir build && cd build
cmake ..
make
./render xxx.pcd ../campose.txt $imgname $imgpath
```