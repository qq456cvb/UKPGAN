#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;


py::array_t<float> estimate_normal(py::array_t<float> pc, double normal_r)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->points.resize(pc.shape(0));
	float *pc_ptr = (float*)pc.request().ptr;
	for (int i = 0; i < pc.shape(0); ++i)
    {
		std::copy(pc_ptr, pc_ptr + 3, &cloud->points[i].data[0]);
		// std::cout << cloud->points[i] << std::endl;
		pc_ptr += 3;
    }
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
//	normalEstimation.setRadiusSearch(normal_r);
	normalEstimation.setKSearch(40);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

	auto result = py::array_t<float>(normals->points.size() * 3);
    auto buf = result.request();
    float *ptr = (float*)buf.ptr;
	for (int i = 0; i < normals->points.size(); ++i)
    {
		std::copy(&normals->points[i].normal[0], &normals->points[i].normal[3], &ptr[i * 3]);
    }
    return result;
}


py::array_t<float> compute(py::array_t<float> pc, double normal_r, double shot_r)
{
	// Object for storing the point cloud.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->points.resize(pc.shape(0));
	float *pc_ptr = (float*)pc.request().ptr;
	for (int i = 0; i < pc.shape(0); ++i)
    {
		std::copy(pc_ptr, pc_ptr + 3, &cloud->points[i].data[0]);
		// std::cout << cloud->points[i] << std::endl;
		pc_ptr += 3;
    }

	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// Object for storing the SHOT descriptors for each point.
	pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>());

	// Note: you would usually perform downsampling now. It has been omitted here
	// for simplicity, but be aware that computation can take a long time.

	// Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
//	normalEstimation.setRadiusSearch(normal_r);
	normalEstimation.setKSearch(40);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

	// SHOT estimation object.
	pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
	shot.setInputCloud(cloud);
	shot.setInputNormals(normals);
	// The radius that defines which of the keypoint's neighbors are described.
	// If too large, there may be clutter, and if too small, not enough points may be found.
	shot.setRadiusSearch(shot_r);
//    shot.setKSearch(40);
	shot.compute(*descriptors);

	auto result = py::array_t<float>(descriptors->points.size() * 352);
    auto buf = result.request();
    float *ptr = (float*)buf.ptr;
    
    for (int i = 0; i < descriptors->points.size(); ++i)
    {
		std::copy(&descriptors->points[i].descriptor[0], &descriptors->points[i].descriptor[352], &ptr[i * 352]);
    }
    return result;
}


PYBIND11_MODULE(shot, m) {
    m.def("compute", &compute, py::arg("pc"), py::arg("normal_r")=0.15, py::arg("shot_r")=0.15);
	m.def("estimate_normal", &estimate_normal, py::arg("pc"), py::arg("normal_r")=0.1);
}