#ifndef VOXBLOX_ROS_OBB_UTILS_H_
#define VOXBLOX_ROS_OBB_UTILS_H_

// Taken from https://codextechnicanum.blogspot.com/2015/04/find-minimum-oriented-bounding-box-of.html

#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

typedef Eigen::Matrix<float, 7, 1> Vector7f;


Vector7f getOBBDetection(
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cluster_cloud,
                    Eigen::Matrix4f post_transformation)
{

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cluster_cloud_post_transform(
                                new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::transformPointCloud (
            *cluster_cloud, *cluster_cloud_post_transform, post_transformation);

    pcl::PointXYZRGBNormal origMinPoint, origMaxPoint;
    pcl::getMinMax3D(*cluster_cloud_post_transform, origMinPoint, origMaxPoint);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud_transformed(
                                            new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t idx = 0u;  idx < (*cluster_cloud_post_transform).size(); ++idx)
    {
      pcl::PointXYZ point;

      float x = cluster_cloud_post_transform->points[idx].x;
      float y = cluster_cloud_post_transform->points[idx].y;

      point.x = x;
      point.y = y;
      point.z = 0.0;

      cluster_cloud_transformed->points.push_back(point);
    }

      // Compute principal directions
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cluster_cloud_transformed, pcaCentroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(
                           *cluster_cloud_transformed, pcaCentroid, covariance);
    Eigen::Matrix2f smallcovariance = covariance.block(0,0,2,2);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigen_solver(
                                   smallcovariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix2f eigenVectorsPCA = eigen_solver.eigenvectors();

    // Transform the original cloud to the origin where the principal
    // components correspond to the axes.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<2,2>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<2,1>(0,3) = -1.f *
                  (projectionTransform.block<2,2>(0,0) * pcaCentroid.head<2>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected(
                                            new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(
        *cluster_cloud_transformed, *cloudPointsProjected, projectionTransform);
    // Get the minimum and maximum points of the transformed cloud.
    pcl::PointXYZ minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    const Eigen::Vector2f meanXY = 0.5f * (maxPoint.getVector3fMap().head<2>() +
                                           minPoint.getVector3fMap().head<2>());
    const float meanZ = 0.5f * (origMaxPoint.z + origMinPoint.z);

    Eigen::Vector2f bboxTransform = eigenVectorsPCA * meanXY +
                                                          pcaCentroid.head<2>();

    Eigen::Vector3f pos(bboxTransform[0], bboxTransform[1] , meanZ);

    float orientation = std::atan2(
                          eigenVectorsPCA.col(0)[1], eigenVectorsPCA.col(0)[0]);
    Eigen::Vector3f dimension(maxPoint.x - minPoint.x,
                              maxPoint.y - minPoint.y,
                              origMaxPoint.z - origMinPoint.z);

    Vector7f detection;
    detection << pos[0], pos[1], pos[2], orientation,
                                      dimension[0], dimension[1], dimension[2];

    return detection;
}

#endif  // VOXBLOX_ROS_OBB_UTILS_H_
