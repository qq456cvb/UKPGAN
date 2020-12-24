/**
3DSmoothNet
main.cpp

Purpose: executes the computation of the SDV voxel grid for the selected interes points

@Author : Zan Gojcic, Caifa Zhou
@Version : 1.0
*/

#include <chrono>
#include <pcl/io/ply_io.h>
#include "core.h"


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

std::vector<std::vector<float>> compute(py::array_t<float, py::array::c_style | py::array::forcecast> pc, float radius, int num_voxels, float smoothing_kernel_width, const std::vector<int>& interest_point_idxs) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud->points.resize(pc.shape(0));
	float *pc_ptr = (float*)pc.request().ptr;
	for (int i = 0; i < pc.shape(0); ++i)
    {
		std::copy(pc_ptr, pc_ptr + 3, &cloud->points[i].data[0]);
		// std::cout << cloud->points[i] << std::endl;
		pc_ptr += 3;
    }

//    std::cout << "Number of Points: " << cloud->size() << std::endl;
//    std::cout << "Size of the voxel grid: " << 2 * radius << std::endl; // Multiplied with two as half size is used (corresponding to the radius)
//    std::cout << "Number of Voxels: " << num_voxels << std::endl;
//    std::cout << "Smoothing Kernel: " << smoothing_kernel_width << std::endl;

    // Specify the parameters of the algorithm
    const int grid_size = num_voxels * num_voxels * num_voxels;
    float voxel_step_size = (2 * radius) / num_voxels;
    float lrf_radius = sqrt(3) * radius; // Such that the circumscribed sphere is obtained

   // Initialize the voxel grid
    flann::Matrix<float> voxel_coordinates = initializeGridMatrix(num_voxels, voxel_step_size, voxel_step_size, voxel_step_size);

    // Compute the local reference frame for all the points
    float smoothing_factor = smoothing_kernel_width * (radius / num_voxels); // Equals half a voxel size so that 3X is 1.5 voxel

    // Check if all the points should be evaluated or only selected  ones
    std::vector<int> evaluation_points;
    if (interest_point_idxs.size() == 0) {
        evaluation_points.resize(cloud->size());
        std::iota(evaluation_points.begin(), evaluation_points.end(), 0);
    } else {
        evaluation_points = interest_point_idxs;
    }

//    std::cout << "Number of keypoints:" << evaluation_points.size() << "\n" << std::endl;

    // Initialize the variables for the NN search and LRF computation
    std::vector<int> indices(cloud->size());
    std::vector<LRF> cloud_lrf(cloud->size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector <std::vector <int>> nearest_neighbors(cloud->size());
    std::vector <std::vector <int>> nearest_neighbors_smoothing(cloud->size());
    std::vector <std::vector <float>> nearest_neighbors_smoothing_dist(cloud->size());

    // Compute the local reference frame for the interes points (code adopted from https://www.researchgate.net/publication/310815969_TOLDI_An_effective_and_robust_approach_for_3D_local_shape_description
    // and not optimized)

    auto t1_lrf = std::chrono::high_resolution_clock::now();
    toldiComputeLRF(cloud, evaluation_points, lrf_radius, 3 * smoothing_factor, cloud_lrf, nearest_neighbors, nearest_neighbors_smoothing, nearest_neighbors_smoothing_dist);
    auto t2_lrf = std::chrono::high_resolution_clock::now();

    // Compute the SDV representation for all the points

    // Start the actuall computation
    auto t1 = std::chrono::high_resolution_clock::now();
    auto features = computeLocalDepthFeature(cloud, evaluation_points, nearest_neighbors, cloud_lrf, radius, voxel_coordinates, num_voxels, smoothing_factor);
    auto t2 = std::chrono::high_resolution_clock::now();

    delete[] voxel_coordinates.ptr();
//    std::cout << "\n---------------------------------------------------------" << std::endl;
//    std::cout << "LRF computation took "
//              << std::chrono::duration_cast<std::chrono::milliseconds>(t2_lrf - t1_lrf).count()
//              << " miliseconds\n";
//    std::cout << "SDV computation took "
//              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
//              << " miliseconds\n";
//    std::cout << "---------------------------------------------------------" << std::endl;

    return features;
}

PYBIND11_MODULE(sdv, m) {
    m.def("compute", &compute, py::arg("pc"), py::arg("radius")=0.15, py::arg("num_voxels")=16, py::arg("num_voxels")=1.75, py::arg("interest_point_idxs")=std::vector<int>());
}
