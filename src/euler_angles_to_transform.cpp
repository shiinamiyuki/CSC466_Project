#include "euler_angles_to_transform.h"
#include <Eigen/Geometry>

Eigen::Affine3d euler_angles_to_transform(
  const Eigen::Vector3d & xzx)
{
  /////////////////////////////////////////////////////////////////////////////
  // Replace with your code
  Eigen::Affine3d A;
 // printf("%lf %lf %lf\n",xzx[0],xzx[1],xzx[2]);
 auto degreesToRadians = [](double x){
     return x / 180.0 * 3.1415926535;
 };
  A = Eigen::AngleAxis<double>(degreesToRadians(xzx[0]),Eigen::Vector3d(1,0,0));
  A *= Eigen::AngleAxis<double>(degreesToRadians(xzx[1]),Eigen::Vector3d(0,0,1));
  A *= Eigen::AngleAxis<double>(degreesToRadians(xzx[2]),Eigen::Vector3d(1,0,0));

  return A;
  /////////////////////////////////////////////////////////////////////////////
}
