#include <Eigen/Dense>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <malloc/_malloc.h>
#include <ostream>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

int main() {
  int x, y, n;
  unsigned char *data = stbi_load("Lenna.png", &x, &y, &n, 0);
  int dataSize = x * y;
  // 226*0.299 +137*0.587 + 125*0.114

  if (data == NULL) {
    std::perror("Failed to Open File!");
  } else {
    printf("%d %d %d\n", x, y, n);
    printf("%d %d %d \n", data[0], data[1], data[2]);
  }

  MatrixXd imgXY(y, x);

  // unsigned char *_data = (unsigned char *)calloc(x * y, sizeof(unsigned
  // char));
  std::vector<int> _data(x * y);
  Eigen::Vector3f greyConversion(0.299, 0.587, 0.114);
  // Eigen::Matrix3d b;

  for (int i = 0; i < dataSize; i++) {
    auto dataRGB =
        Eigen::Vector3f(data[i * n], data[i * n + 1], data[i * n + 2]);
    int greyCode = (int)dataRGB.dot(greyConversion);
    _data[i] = greyCode;
    // std::cout << i / y << " " << i % x << " " << greyCode << std::endl;
    int row = i / x;
    int col = i % x;
    imgXY(col, row) = greyCode;
  }

  // BDCSVD<MatrixXd> svd(mf, ComputeFullU | ComputeFullV);

  BDCSVD<MatrixXd, ComputeFullU | ComputeFullV> svd(imgXY);

  auto U = svd.matrixU();
  auto V = svd.matrixV();
  auto singularValues = svd.singularValues();

  // Eigen::VectorXd singularValues = svd.singularValues();

  // // for (auto it = singularValues.end() - 120; it != singularValues.end();
  // // it++) {
  // //   *it = 0;
  // // }
  // MatrixXd cImgMat =
  //     U * singularValues.asDiagonal().toDenseMatrix() * V.transpose();

  MatrixXd cImgMat = U * singularValues.asDiagonal() * V.transpose();
  std::vector<double> imgVec(cImgMat.size());
  Map<MatrixXd>(imgVec.data(), cImgMat.rows(), cImgMat.cols()) = cImgMat;

  std::vector<unsigned char> charVec(imgVec.begin(), imgVec.end());

  // for (int i = 0; i < imgVec.size(); i++) {
  // std::cout << (int)imgVec.at(i) << std::endl;
  // _data[i] = (int)imgVec.at(i);
  // }

  stbi_write_png("LennaBW.png", x, y, 1, charVec.data(),
                 sizeof(unsigned char) * n);

  // // std::ofstream outFile("my_file.txt");

  // // for (const auto &s : singVals)
  // //   outFile << s << "\n";

  // free(_data);
  stbi_image_free(data);
}