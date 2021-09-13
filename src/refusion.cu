// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
// Part of the code in this file was adapted from the original VoxelHashing
// implementation by Niessner et al.
// https://github.com/niessner/VoxelHashing/blob/master/DepthSensingCUDA/Source/cuda_SimpleMatrixUtil.h

// MESH_EXTRACTOR.CU -----------------------------------------------------------
#include "marching_cubes/mesh_extractor.h"
#include "utils/utils.h"
#include "marching_cubes/lookup_tables.h"

// MESH.CU ---------------------------------------------------------------------
#include "marching_cubes/mesh.h"
#include <algorithm>

// TRACER.CU -------------------------------------------------------------------
#include "tracker/tracker.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include "tracker/eigen_wrapper.h"
#include "utils/matrix_utils.h"
#include "utils/utils.h"
#include "utils/rgbd_image.h"

// HASH_TABLE.CU ---------------------------------------------------------------
#include "tsdfvh/hash_table.h"
#include <iostream>

// HEAP.CU ---------------------------------------------------------------------
#include "tsdfvh/heap.h"

// TSDF_VOLUME.CU --------------------------------------------------------------
#include "tsdfvh/tsdf_volume.h"
#include <cfloat>
#include <cmath>
#include "marching_cubes/mesh_extractor.h"

// EigenWrapper dependency removal
#include <Eigen/Core>
#include <tracker/eigen_wrapper.h>
#include <Eigen/Cholesky>
#include <unsupported/Eigen/MatrixFunctions>

#include <cstring>

#define THREADS_PER_BLOCK 512
#define THREADS_PER_BLOCK2 64
#define THREADS_PER_BLOCK3 32

namespace refusion {
namespace tsdfvh {

// MESH_EXTRACTOR.CU -----------------------------------------------------------

void MeshExtractor::Init(unsigned int max_triangles, float voxel_size) {
  mesh_ = new Mesh;
  mesh_->Init(max_triangles);
  voxel_size_ = voxel_size;
}

void MeshExtractor::Free() {
  mesh_->Free();
  delete mesh_;
}

__device__ bool TrilinearInterpolation(TsdfVolume *volume, float voxel_size,
                                       const float3 &position, float &distance,
                                       uchar3 &color, int i, unsigned int* calls) {
  const float3 pos_dual =
      position -
      make_float3(voxel_size / 2.0f, voxel_size / 2.0f, voxel_size / 2.0f);
  float3 voxel_position = position / voxel_size;
  float3 weight = make_float3(voxel_position.x - floor(voxel_position.x),
                              voxel_position.y - floor(voxel_position.y),
                              voxel_position.z - floor(voxel_position.z));

  distance = 0.0f;
  float3 color_float = make_float3(0.0f, 0.0f, 0.0f);

  Voxel v = volume->GetVoxel(pos_dual + make_float3(0.0f, 0.0f, 0.0f), i, calls);
  if (v.weight == 0) return false;
  float3 vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
  color_float =
      color_float +
      (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(voxel_size, 0.0f, 0.0f), i, calls);
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += weight.x * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
  color_float =
      color_float + weight.x * (1.0f - weight.y) * (1.0f - weight.z) * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(0.0f, voxel_size, 0.0f), i, calls);
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += (1.0f - weight.x) * weight.y * (1.0f - weight.z) * v.sdf;
  color_float =
      color_float + (1.0f - weight.x) * weight.y * (1.0f - weight.z) * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(0.0f, 0.0f, voxel_size), i, calls);
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += (1.0f - weight.x) * (1.0f - weight.y) * weight.z * v.sdf;
  color_float =
      color_float + (1.0f - weight.x) * (1.0f - weight.y) * weight.z * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(voxel_size, voxel_size, 0.0f), i, calls);
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += weight.x * weight.y * (1.0f - weight.z) * v.sdf;
  color_float = color_float + weight.x * weight.y * (1.0f - weight.z) * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(0.0f, voxel_size, voxel_size), i, calls);
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += (1.0f - weight.x) * weight.y * weight.z * v.sdf;
  color_float = color_float + (1.0f - weight.x) * weight.y * weight.z * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(voxel_size, 0.0f, voxel_size), i, calls);
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += weight.x * (1.0f - weight.y) * weight.z * v.sdf;
  color_float = color_float + weight.x * (1.0f - weight.y) * weight.z * vColor;

  v = volume->GetVoxel(pos_dual +
                       make_float3(voxel_size, voxel_size, voxel_size), i, calls);
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += weight.x * weight.y * weight.z * v.sdf;
  color_float = color_float + weight.x * weight.y * weight.z * vColor;

  color = make_uchar3(color_float.x, color_float.y, color_float.z);

  return true;
}

__device__ Vertex VertexInterpolation(float isolevel, const float3 &p1,
                                      const float3 &p2, float d1, float d2,
                                      const uchar3 &c1, const uchar3 &c2) {
  Vertex r1; r1.position = p1; r1.color = make_float3(c1.x, c1.y, c1.z) / 255.f;
  Vertex r2; r2.position = p2; r2.color = make_float3(c2.x, c2.y, c2.z) / 255.f;

  if (fabs(isolevel - d1) < 0.00001f) return r1;
  if (fabs(isolevel - d2) < 0.00001f) return r2;
  if (fabs(d1 - d2) < 0.00001f) return r1;

  float mu = (isolevel - d1) / (d2 - d1);

  Vertex res;
  // Position
  res.position.x = p1.x + mu * (p2.x - p1.x);
  res.position.y = p1.y + mu * (p2.y - p1.y);
  res.position.z = p1.z + mu * (p2.z - p1.z);

  // Color
  res.color.x =
      static_cast<float>(c1.x + mu * static_cast<float>(c2.x - c1.x)) / 255.f;
  res.color.y =
      static_cast<float>(c1.y + mu * static_cast<float>(c2.y - c1.y)) / 255.f;
  res.color.z =
      static_cast<float>(c1.z + mu * static_cast<float>(c2.z - c1.z)) / 255.f;

  return res;
}

__device__ void ExtractMeshAtPosition(TsdfVolume *volume,
                                      const float3 &position, float voxel_size,
                                      Mesh *mesh, int i, unsigned int* calls) {
  const float isolevel = 0.0f;
  const float P = voxel_size/2.0f;
  const float M = -P;

  float3 p000 = position + make_float3(M, M, M);
  float dist000;
  uchar3 color000;
  if (!TrilinearInterpolation(volume, voxel_size, p000, dist000, color000, i, calls))
    return;

  float3 p100 = position + make_float3(P, M, M);
  float dist100;
  uchar3 color100;
  if (!TrilinearInterpolation(volume, voxel_size, p100, dist100, color100, i, calls))
    return;

  float3 p010 = position + make_float3(M, P, M);
  float dist010;
  uchar3 color010;
  if (!TrilinearInterpolation(volume, voxel_size, p010, dist010, color010, i, calls))
    return;

  float3 p001 = position + make_float3(M, M, P);
  float dist001;
  uchar3 color001;
  if (!TrilinearInterpolation(volume, voxel_size, p001, dist001, color001, i, calls))
    return;

  float3 p110 = position + make_float3(P, P, M);
  float dist110;
  uchar3 color110;
  if (!TrilinearInterpolation(volume, voxel_size, p110, dist110, color110, i, calls))
    return;

  float3 p011 = position + make_float3(M, P, P);
  float dist011;
  uchar3 color011;
  if (!TrilinearInterpolation(volume, voxel_size, p011, dist011, color011, i, calls))
    return;

  float3 p101 = position + make_float3(P, M, P);
  float dist101;
  uchar3 color101;
  if (!TrilinearInterpolation(volume, voxel_size, p101, dist101, color101, i, calls))
    return;

  float3 p111 = position + make_float3(P, P, P);
  float dist111;
  uchar3 color111;
  if (!TrilinearInterpolation(volume, voxel_size, p111, dist111, color111, i, calls))
    return;

  uint cubeindex = 0;
  if (dist010 < isolevel) cubeindex += 1;
  if (dist110 < isolevel) cubeindex += 2;
  if (dist100 < isolevel) cubeindex += 4;
  if (dist000 < isolevel) cubeindex += 8;
  if (dist011 < isolevel) cubeindex += 16;
  if (dist111 < isolevel) cubeindex += 32;
  if (dist101 < isolevel) cubeindex += 64;
  if (dist001 < isolevel) cubeindex += 128;

  if (edgeTable[cubeindex] == 0) return;

  Voxel v = volume->GetVoxel(position, i, calls);

  Vertex vertlist[12];
  if (edgeTable[cubeindex] & 1)
    vertlist[0] = VertexInterpolation(isolevel, p010, p110, dist010, dist110,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 2)
    vertlist[1] = VertexInterpolation(isolevel, p110, p100, dist110, dist100,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 4)
    vertlist[2] = VertexInterpolation(isolevel, p100, p000, dist100, dist000,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 8)
    vertlist[3] = VertexInterpolation(isolevel, p000, p010, dist000, dist010,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 16)
    vertlist[4] = VertexInterpolation(isolevel, p011, p111, dist011, dist111,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 32)
    vertlist[5] = VertexInterpolation(isolevel, p111, p101, dist111, dist101,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 64)
    vertlist[6] = VertexInterpolation(isolevel, p101, p001, dist101, dist001,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 128)
    vertlist[7] = VertexInterpolation(isolevel, p001, p011, dist001, dist011,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 256)
    vertlist[8] = VertexInterpolation(isolevel, p010, p011, dist010, dist011,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 512)
    vertlist[9] = VertexInterpolation(isolevel, p110, p111, dist110, dist111,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 1024)
    vertlist[10] = VertexInterpolation(isolevel, p100, p101, dist100, dist101,
                                       v.color, v.color);
  if (edgeTable[cubeindex] & 2048)
    vertlist[11] = VertexInterpolation(isolevel, p000, p001, dist000, dist001,
                                       v.color, v.color);

  for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
    Triangle t;
    t.v0 = vertlist[triTable[cubeindex][i + 0]];
    t.v1 = vertlist[triTable[cubeindex][i + 1]];
    t.v2 = vertlist[triTable[cubeindex][i + 2]];

    mesh->AppendTriangle(t);
  }
}

__global__ void ExtractMeshKernel(TsdfVolume *volume, float3 lower_corner,
                                  float3 upper_corner, float voxel_size,
                                  Mesh *mesh, unsigned int* calls) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float3 size = lower_corner - upper_corner;
  size = make_float3(fabs(size.x), fabs(size.y), fabs(size.z));
  int3 grid_size =
      make_int3(size.x / voxel_size, size.y / voxel_size, size.z / voxel_size);
  int grid_linear_size = grid_size.x * grid_size.y * grid_size.z;
  for (int i = index; i < grid_linear_size; i += stride) {
    // Delinearize index
    int3 grid_position =
        make_int3(i % grid_size.x, (i / grid_size.x) % grid_size.y,
                  i / (grid_size.x * grid_size.y));
    float3 world_position = make_float3(
        static_cast<float>(grid_position.x) * voxel_size + lower_corner.x,
        static_cast<float>(grid_position.y) * voxel_size + lower_corner.y,
        static_cast<float>(grid_position.z) * voxel_size + lower_corner.z);
    ExtractMeshAtPosition(volume, world_position, voxel_size, mesh, i, calls);
  }
}

void MeshExtractor::ExtractMesh(TsdfVolume *volume, float3 lower_corner,
                                float3 upper_corner) {
  int threads_per_block = 256;
  int thread_blocks =
      (volume->GetOptions().num_blocks * volume->GetOptions().block_size +
       threads_per_block - 1) /
      threads_per_block;

  std::size_t mesh_size = sizeof(Mesh);
  std::size_t triangles_size = sizeof(Triangle) * mesh_->max_triangles_;

  Mesh* mesh_d;
  Triangle* triangles_d;

  cudaMalloc(&mesh_d, mesh_size);
  cudaMalloc(&triangles_d, triangles_size);
  cudaMemcpy(mesh_d, mesh_, mesh_size, cudaMemcpyHostToDevice);
  cudaMemcpy(triangles_d, mesh_->triangles_, triangles_size, cudaMemcpyHostToDevice);

  cudaMemcpy(&mesh_d->triangles_, &triangles_d, sizeof(Triangle*), cudaMemcpyHostToDevice);

  const unsigned int total_threads = thread_blocks * threads_per_block;
  unsigned int* cache_calls_h = new unsigned int[total_threads];
  unsigned int* cache_calls_d;
  cudaMalloc(&cache_calls_d, total_threads * sizeof(unsigned int));
  cudaMemcpy(cache_calls_d, 0, total_threads * sizeof(unsigned int), cudaMemcpyHostToDevice);

  ExtractMeshKernel<<<thread_blocks, threads_per_block>>>(
      volume, lower_corner, upper_corner, voxel_size_, mesh_, cache_calls_d);
  cudaDeviceSynchronize();

  cudaMemcpy(cache_calls_h, cache_calls_d, total_threads * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(cache_calls_d);

  Triangle* triangles = mesh_->triangles_;

  cudaMemcpy(triangles, triangles_d, triangles_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(mesh_, mesh_d, mesh_size, cudaMemcpyDeviceToHost);
  mesh_->triangles_ = triangles;

  cudaFree(triangles_d);
  cudaFree(mesh_d);
}

Mesh MeshExtractor::GetMesh() {
  return *mesh_;
}

// MESH.CU ---------------------------------------------------------------------

void Mesh::Init(unsigned int max_triangles) {
  num_triangles_ = 0;
  max_triangles_ = max_triangles;
  triangles_ = new Triangle[max_triangles_];
}

void Mesh::Free() {
  delete[] triangles_;
}

__device__ void Mesh::AppendTriangle(Triangle t) {
  unsigned int idx = atomicAdd(&num_triangles_, 1);
  if (num_triangles_ <= max_triangles_) triangles_[idx] = t;
}

void Mesh::SaveToFile(const std::string &filename) {
  std::ofstream fout(filename);
  int n = std::min(num_triangles_, max_triangles_);
  if (n == max_triangles_) {
    std::cout << "Triangles limit reached!" << std::endl;
  }
  for (unsigned int i = 0; i < n; i++) {
    fout << "v " << triangles_[i].v0.position.x << " "
         << triangles_[i].v0.position.y << " " << triangles_[i].v0.position.z
         << " " << triangles_[i].v0.color.x << " " << triangles_[i].v0.color.y
         << " " << triangles_[i].v0.color.z << std::endl;
    fout << "v " << triangles_[i].v1.position.x << " "
         << triangles_[i].v1.position.y << " " << triangles_[i].v1.position.z
         << " " << triangles_[i].v1.color.x << " " << triangles_[i].v1.color.y
         << " " << triangles_[i].v1.color.z << std::endl;
    fout << "v " << triangles_[i].v2.position.x << " "
         << triangles_[i].v2.position.y << " " << triangles_[i].v2.position.z
         << " " << triangles_[i].v2.color.x << " " << triangles_[i].v2.color.y
         << " " << triangles_[i].v2.color.z << std::endl;
  }
  for (unsigned int i = 1; i <= n * 3; i += 3) {
    fout << "f " << i << " " << i + 1 << " " << i + 2 << std::endl;
  }
  fout.close();
}

// HASH_TABLE.CU ---------------------------------------------------------------

__global__ void InitEntriesKernel(HashEntry *entries, int num_entries) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < num_entries; i += stride) {
    entries[i].pointer = kFreeEntry;
    entries[i].position = make_int3(0, 0, 0);
  }
}

__global__ void InitHeapKernel(Heap *heap, VoxelBlock *voxel_blocks,
                               int num_blocks, int block_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (index == 0) {
    heap->heap_counter_ = num_blocks - 1;
  }

  for (int i = index; i < num_blocks; i += stride) {
    heap->heap_[i] = num_blocks - i - 1;
  }
}

void HashTable::Init(int num_buckets, int bucket_size, int num_blocks,
                     int block_size) {
  num_buckets_ = num_buckets;
  bucket_size_ = bucket_size;
  num_entries_ = num_buckets * bucket_size;
  num_blocks_ = num_blocks;
  block_size_ = block_size;
  num_allocated_blocks_ = 0;

  int block_size_3 = block_size * block_size * block_size;
  int voxel_size = sizeof(Voxel) * block_size_3 * num_blocks;
  cudaMalloc(&entries_, sizeof(HashEntry) * num_entries_);
  cudaMalloc(&voxels_, voxel_size);

  std::size_t voxel_blocks_size = sizeof(VoxelBlock) * num_blocks;

  VoxelBlock* voxel_blocks = new VoxelBlock[num_blocks];

  cudaMalloc(&voxel_blocks_, voxel_blocks_size);

  Heap* heap = new Heap;

  cudaMalloc(&heap_, sizeof(Heap));
  cudaDeviceSynchronize();
  for (size_t i = 0; i < num_blocks; i++) {
    voxel_blocks[i].Init(&(voxels_[i * block_size * block_size * block_size]),
                          block_size);
  }

  cudaMemcpy(voxel_blocks_, voxel_blocks, voxel_blocks_size, cudaMemcpyHostToDevice);  
  delete[] voxel_blocks;

  heap->Init(num_blocks);
  cudaMemcpy(heap_, heap, sizeof(Heap), cudaMemcpyHostToDevice);

  delete heap;

  int threads_per_block = THREADS_PER_BLOCK;
  int thread_blocks =
      (num_entries_ + threads_per_block - 1) / threads_per_block;
  InitEntriesKernel<<<thread_blocks, threads_per_block>>>(entries_,
                                                          num_entries_);
  cudaDeviceSynchronize();

  thread_blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
  InitHeapKernel<<<thread_blocks, threads_per_block>>>(heap_, voxel_blocks_,
                                                       num_blocks, block_size);
  cudaDeviceSynchronize();

  cudaMemset(voxels_, 0, voxel_size);
  cudaDeviceSynchronize();
}

void HashTable::Free() {
  cudaFree(entries_);
  cudaFree(voxels_);
  cudaFree(voxel_blocks_);
  cudaFree(heap_);
}

__device__ int HashTable::AllocateBlock(const int3 &position, unsigned int* heap_counter) {
  int bucket_idx = Hash(position);

  int free_entry_idx = -1;
  for (int i = 0; i < bucket_size_; i++) {
    if (entries_[bucket_idx + i].position.x == position.x &&
        entries_[bucket_idx + i].position.y == position.y &&
        entries_[bucket_idx + i].position.z == position.z &&
        entries_[bucket_idx + i].pointer != kFreeEntry) {
      return 0;
    }
    if (free_entry_idx == -1 &&
        entries_[bucket_idx + i].pointer == kFreeEntry) {
      free_entry_idx = bucket_idx + i;
    }
  }

  if (free_entry_idx != -1) {
    int mutex = 0;
    mutex =
        atomicCAS(&entries_[free_entry_idx].pointer, kFreeEntry, kLockEntry);
    if (mutex == kFreeEntry) {
      entries_[free_entry_idx].position = position;
      *heap_counter = atomicSub(heap_counter, 1);
      entries_[free_entry_idx].pointer = 500000 - *heap_counter - 1;
      atomicAdd(&num_allocated_blocks_, 1);
      return 1;
    }
  }
  return -1;
}

__device__ bool HashTable::DeleteBlock(
    const int3 &position) {
  int bucket_idx = Hash(position);

  for (int i = 0; i < bucket_size_; i++) {
    if (entries_[bucket_idx + i].position.x == position.x &&
        entries_[bucket_idx + i].position.y == position.y &&
        entries_[bucket_idx + i].position.z == position.z &&
        entries_[bucket_idx + i].pointer != kFreeEntry) {
      int ptr = entries_[bucket_idx + i].pointer;
      for(int j=0;j<block_size_ * block_size_ * block_size_; j++) {
        voxel_blocks_[ptr].at(j).sdf = 0;
        voxel_blocks_[ptr].at(j).color = make_uchar3(0, 0, 0);
        voxel_blocks_[ptr].at(j).weight = 0;
      }
      heap_->Append(ptr);
      entries_[bucket_idx + i].pointer = kFreeEntry;
      entries_[bucket_idx + i].position = make_int3(0, 0, 0);
      return true;
    }
  }
  return false;
}

__host__ __device__ HashEntry HashTable::FindHashEntry(int3 position) {
  int bucket_idx = Hash(position);
  for (int i = 0; i < bucket_size_; i++) {
    if (entries_[bucket_idx + i].position.x == position.x &&
        entries_[bucket_idx + i].position.y == position.y &&
        entries_[bucket_idx + i].position.z == position.z &&
        entries_[bucket_idx + i].pointer != kFreeEntry) {
      return entries_[bucket_idx + i];
    }
  }
  HashEntry entry;
  entry.position = position;
  entry.pointer = kFreeEntry;
  return entry;
}

__host__ __device__ int HashTable::Hash(int3 position) {
  const int p1 = 73856093;
  const int p2 = 19349669;
  const int p3 = 83492791;

  int result = ((position.x * p1) ^ (position.y * p2) ^ (position.z * p3)) %
               num_buckets_;
  if (result < 0) {
    result += num_buckets_;
  }
  return result * bucket_size_;
}

int HashTable::GetNumAllocatedBlocks() {
  return num_allocated_blocks_;
}

__host__ __device__ int HashTable::GetNumEntries() {
  return num_entries_;
}

__host__ __device__ HashEntry HashTable::GetHashEntry(int i) {
  return entries_[i];
}

// HEAP.CU ---------------------------------------------------------------------

void Heap::Init(int heap_size) {
  cudaMalloc(&heap_, sizeof(unsigned int) * heap_size);
}

// __device__ unsigned int Heap::Consume() {
//   unsigned int idx = atomicSub(&heap_counter_, 1);
//   return heap_[idx];
// }

__device__ void Heap::Append(unsigned int ptr) {
  unsigned int idx = atomicAdd(&heap_counter_, 1);
  heap_[idx + 1] = ptr;
}

// TSDF_VOLUME.CU --------------------------------------------------------------

void TsdfVolume::Init(const TsdfVolumeOptions &options) {
  options_ = options;
  HashTable::Init(options_.num_buckets, options_.bucket_size,
                  options_.num_blocks, options_.block_size);
}

void TsdfVolume::Free() { HashTable::Free(); }

__host__ __device__ float3 TsdfVolume::GlobalVoxelToWorld(int3 position) {
  return make_float3(position.x * options_.voxel_size,
                     position.y * options_.voxel_size,
                     position.z * options_.voxel_size);
}

__host__ __device__ int3 TsdfVolume::WorldToGlobalVoxel(float3 position) {
  return make_int3(position.x / options_.voxel_size + signf(position.x) * 0.5f,
                   position.y / options_.voxel_size + signf(position.y) * 0.5f,
                   position.z / options_.voxel_size + signf(position.z) * 0.5f);
}

__host__ __device__ int3 TsdfVolume::WorldToBlock(float3 position) {
  int3 voxel_position = WorldToGlobalVoxel(position);
  int3 block_position;
  if (voxel_position.x < 0)
    block_position.x = (voxel_position.x - block_size_ + 1) / block_size_;
  else
    block_position.x = voxel_position.x / block_size_;

  if (voxel_position.y < 0)
    block_position.y = (voxel_position.y - block_size_ + 1) / block_size_;
  else
    block_position.y = voxel_position.y / block_size_;

  if (voxel_position.z < 0)
    block_position.z = (voxel_position.z - block_size_ + 1) / block_size_;
  else
    block_position.z = voxel_position.z / block_size_;

  return block_position;
}

__host__ __device__ int3 TsdfVolume::WorldToLocalVoxel(float3 position) {
  int3 position_global = WorldToGlobalVoxel(position);
  int3 position_local = make_int3(position_global.x % block_size_,
                                  position_global.y % block_size_,
                                  position_global.z % block_size_);
  if (position_local.x < 0) position_local.x += block_size_;
  if (position_local.y < 0) position_local.y += block_size_;
  if (position_local.z < 0) position_local.z += block_size_;
  return position_local;
}

__host__ __device__ Voxel TsdfVolume::GetVoxel(float3 position, int i, unsigned int* calls) {
  int3 block_position = WorldToBlock(position);
  int3 local_voxel = WorldToLocalVoxel(position);

  int entry_pos = 0;
#ifdef __CUDA_ARCH__
  asm volatile("mad24.lo.s32 %0, %1, %2, %3;"
               : "=r"(entry_pos)
               : "r"(block_position.x),
                 "r"(block_position.y),
                 "r"(block_position.z));

  bool cache_miss;
  if (cache_miss = (entry_pos == 0))
#endif
    entry_pos = HashTable::FindHashEntry(block_position).pointer;

  Voxel voxel;
  if (entry_pos != kFreeEntry)
    voxel = HashTable::voxel_blocks_[entry_pos].at(local_voxel);
  else {
    calls[i]++;

    voxel.sdf = 0;
    voxel.color = make_uchar3(0, 0, 0);
    voxel.weight = 0;
  }

#ifdef __CUDA_ARCH__
  if (cache_miss) {
    asm volatile("bfi.b32 %0, %1, %2, %3, %4;"
                 : "=r"(entry_pos)
                 : "r"(block_position.x),
                   "r"(block_position.y),
                   "r"(block_position.z),
                   "r"(entry_pos));
  }
#endif

  return voxel;
}

__host__ __device__ Voxel TsdfVolume::GetInterpolatedVoxel(float3 position, int i, unsigned int* calls) {
  Voxel v0 = GetVoxel(position, i, calls);
  if (v0.weight == 0) return v0;
  float voxel_size = options_.voxel_size;
  const float3 pos_dual =
      position -
      make_float3(voxel_size / 2.0f, voxel_size / 2.0f, voxel_size / 2.0f);
  float3 voxel_position = position / voxel_size;
  float3 weight = make_float3(voxel_position.x - floor(voxel_position.x),
                              voxel_position.y - floor(voxel_position.y),
                              voxel_position.z - floor(voxel_position.z));

  float distance = 0.0f;
  float3 color_float = make_float3(0.0f, 0.0f, 0.0f);
  float3 vColor;

  Voxel v = GetVoxel(pos_dual + make_float3(0.0f, 0.0f, 0.0f), i, calls);
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance +=
        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * v0.sdf;
    color_float =
        color_float +
        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance +=
        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
    color_float =
        color_float +
        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(voxel_size, 0.0f, 0.0f), i, calls);
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += weight.x * (1.0f - weight.y) * (1.0f - weight.z) * v0.sdf;
    color_float =
        color_float + weight.x * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += weight.x * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
    color_float =
        color_float + weight.x * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(0.0f, voxel_size, 0.0f), i, calls);
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += (1.0f - weight.x) * weight.y * (1.0f - weight.z) * v0.sdf;
    color_float =
        color_float + (1.0f - weight.x) * weight.y * (1.0f - weight.z) * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += (1.0f - weight.x) * weight.y * (1.0f - weight.z) * v.sdf;
    color_float =
        color_float + (1.0f - weight.x) * weight.y * (1.0f - weight.z) * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(0.0f, 0.0f, voxel_size), i, calls);
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += (1.0f - weight.x) * (1.0f - weight.y) * weight.z * v0.sdf;
    color_float =
        color_float + (1.0f - weight.x) * (1.0f - weight.y) * weight.z * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += (1.0f - weight.x) * (1.0f - weight.y) * weight.z * v.sdf;
    color_float =
        color_float + (1.0f - weight.x) * (1.0f - weight.y) * weight.z * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(voxel_size, voxel_size, 0.0f), i, calls);
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += weight.x * weight.y * (1.0f - weight.z) * v0.sdf;
    color_float =
        color_float + weight.x * weight.y * (1.0f - weight.z) * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += weight.x * weight.y * (1.0f - weight.z) * v.sdf;
    color_float =
        color_float + weight.x * weight.y * (1.0f - weight.z) * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(0.0f, voxel_size, voxel_size), i, calls);
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += (1.0f - weight.x) * weight.y * weight.z * v0.sdf;
    color_float =
        color_float + (1.0f - weight.x) * weight.y * weight.z * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += (1.0f - weight.x) * weight.y * weight.z * v.sdf;
    color_float =
        color_float + (1.0f - weight.x) * weight.y * weight.z * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(voxel_size, 0.0f, voxel_size), i, calls);
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += weight.x * (1.0f - weight.y) * weight.z * v0.sdf;
    color_float =
        color_float + weight.x * (1.0f - weight.y) * weight.z * vColor;
  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += weight.x * (1.0f - weight.y) * weight.z * v.sdf;
    color_float =
        color_float + weight.x * (1.0f - weight.y) * weight.z * vColor;
  }

  v = GetVoxel(pos_dual + make_float3(voxel_size, voxel_size, voxel_size), i, calls);
  if (v.weight == 0) {
    vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
    distance += weight.x * weight.y * weight.z * v0.sdf;
    color_float = color_float + weight.x * weight.y * weight.z * vColor;

  } else {
    vColor = make_float3(v.color.x, v.color.y, v.color.z);
    distance += weight.x * weight.y * weight.z * v.sdf;
    color_float = color_float + weight.x * weight.y * weight.z * vColor;
  }

  uchar3 color = make_uchar3(color_float.x, color_float.y, color_float.z);
  v.weight = v0.weight;
  v.sdf = distance;
  v.color = color;
  return v;
}

__host__ __device__ bool TsdfVolume::SetVoxel(float3 position,
                                              const Voxel &voxel) {
  int3 block_position = WorldToBlock(position);
  int3 local_voxel = WorldToLocalVoxel(position);
  HashEntry entry = HashTable::FindHashEntry(block_position);
  if (entry.pointer == kFreeEntry) {
    return false;
  }
  HashTable::voxel_blocks_[entry.pointer].at(local_voxel) = voxel;
  return true;
}

__host__ __device__ bool TsdfVolume::UpdateVoxel(float3 position,
                                                 const Voxel &voxel) {
  int3 block_position = WorldToBlock(position);
  int3 local_voxel = WorldToLocalVoxel(position);
  HashEntry entry = HashTable::FindHashEntry(block_position);
  if (entry.pointer == kFreeEntry) {
    return false;
  }
  HashTable::voxel_blocks_[entry.pointer]
      .at(local_voxel)
      .Combine(voxel, options_.max_sdf_weight);
  return true;
}

__global__ void AllocateFromDepthKernel(TsdfVolume *volume, float *depth,
                                        RgbdSensor sensor, float4x4 transform,
                                        unsigned int* heap_counter) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size = sensor.rows * sensor.cols;

  float truncation_distance = volume->GetOptions().truncation_distance;
  float block_size =
      volume->GetOptions().block_size * volume->GetOptions().voxel_size;

  float3 start_pt = make_float3(transform.m14, transform.m24, transform.m34);
  for (int i = index; i < size; i += stride) {
    if (depth[i] < volume->GetOptions().min_sensor_depth ||
        depth[i] > volume->GetOptions().max_sensor_depth)
      continue;
    float3 point = GetPoint3d(i, depth[i], sensor);
    point = transform * point;
    if (point.x == 0 && point.y == 0 && point.z == 0) continue;
    // compute start and end of the ray
    float3 ray_direction = normalize(point - start_pt);
    float surface_distance = distance(start_pt, point);
    float3 ray_start = start_pt;
    float3 ray_end =
        start_pt + ray_direction * (surface_distance + truncation_distance);
    // traverse the ray discretely using the block size and allocate it
    // adapted from https://github.com/francisengelmann/fast_voxel_traversal/blob/master/main.cpp
    int3 block_start = make_int3(floor(ray_start.x / block_size),
                                 floor(ray_start.y / block_size),
                                 floor(ray_start.z / block_size));

    int3 block_end = make_int3(floor(ray_end.x / block_size),
                               floor(ray_end.y / block_size),
                               floor(ray_end.z / block_size));

    int3 block_position = block_start;
    int3 step = make_int3(sign(ray_direction.x),
                          sign(ray_direction.y),
                          sign(ray_direction.z));

    float3 delta_t;
    delta_t.x =
        (ray_direction.x != 0) ? fabs(block_size / ray_direction.x) : FLT_MAX;
    delta_t.y =
        (ray_direction.y != 0) ? fabs(block_size / ray_direction.y) : FLT_MAX;
    delta_t.z =
        (ray_direction.z != 0) ? fabs(block_size / ray_direction.z) : FLT_MAX;

    float3 boundary = make_float3(
        (block_position.x + static_cast<float>(step.x)) * block_size,
        (block_position.y + static_cast<float>(step.y)) * block_size,
        (block_position.z + static_cast<float>(step.z)) * block_size);

    float3 max_t;
    max_t.x = (ray_direction.x != 0)
                  ? (boundary.x - ray_start.x) / ray_direction.x
                  : FLT_MAX;
    max_t.y = (ray_direction.y != 0)
                  ? (boundary.y - ray_start.y) / ray_direction.y
                  : FLT_MAX;
    max_t.z = (ray_direction.z != 0)
                  ? (boundary.z - ray_start.z) / ray_direction.z
                  : FLT_MAX;

    int3 diff = make_int3(0, 0, 0);
    bool neg_ray = false;

    if (block_position.x != block_end.x && ray_direction.x < 0) {
      diff.x--;
      neg_ray = true;
    }
    if (block_position.y != block_end.y && ray_direction.y < 0) {
      diff.y--;
      neg_ray = true;
    }
    if (block_position.z != block_end.z && ray_direction.z < 0) {
      diff.z--;
      neg_ray = true;
    }
    volume->AllocateBlock(block_position, heap_counter);

    if (neg_ray) {
      block_position = block_position + diff;
      volume->AllocateBlock(block_position, heap_counter);
    }

    while (block_position.x != block_end.x || block_position.y != block_end.y ||
           block_position.z != block_end.z) {
      if (max_t.x < max_t.y) {
        if (max_t.x < max_t.z) {
          block_position.x += step.x;
          max_t.x += delta_t.x;
        } else {
          block_position.z += step.z;
          max_t.z += delta_t.z;
        }
      } else {
        if (max_t.y < max_t.z) {
          block_position.y += step.y;
          max_t.y += delta_t.y;
        } else {
          block_position.z += step.z;
          max_t.z += delta_t.z;
        }
      }
      volume->AllocateBlock(block_position, heap_counter);
    }
  }
}

__global__ void IntegrateScanKernel(TsdfVolume *volume, uchar3 *color,
                                    float *depth, RgbdSensor sensor,
                                    float4x4 transform, float4x4 inv_transform,
                                    bool *mask) {
  //loop through ALL entries
  //  if entry is in camera frustum
  //    loop through voxels inside block
    //    if voxel is in truncation region
    //      update voxels
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int block_size = volume->GetOptions().block_size;
  float voxel_size = volume->GetOptions().voxel_size;
  float truncation_distance = volume->GetOptions().truncation_distance;

  for (int i = index; i < volume->GetNumEntries(); i += stride) {
    float3 position = make_float3(
        volume->GetHashEntry(i).position.x * voxel_size * block_size,
        volume->GetHashEntry(i).position.y * voxel_size * block_size,
        volume->GetHashEntry(i).position.z * voxel_size * block_size);
    // To camera coordinates
    float3 position_cam = inv_transform * position;
    // If behind camera plane discard
    if (position_cam.z < 0) continue;
    float3 block_center =
        make_float3(position_cam.x + 0.5 * voxel_size * block_size,
                    position_cam.y + 0.5 * voxel_size * block_size,
                    position_cam.z + 0.5 * voxel_size * block_size);
    int2 image_position = Project(block_center, sensor);
    if (image_position.x >= 0 && image_position.y >= 0 &&
        image_position.x < sensor.cols && image_position.y < sensor.rows) {
      float3 start_pt = make_float3(0, 0, 0);

      for (int bx = 0; bx < block_size; bx++) {
        for (int by = 0; by < block_size; by++) {
          for (int bz = 0; bz < block_size; bz++) {
            float3 voxel_position = make_float3(position.x + bx * voxel_size,
                                                position.y + by * voxel_size,
                                                position.z + bz * voxel_size);
            voxel_position = inv_transform * voxel_position;
            image_position = Project(voxel_position, sensor);
            // Check again inside the block
            if (image_position.x >= 0 && image_position.y >= 0 &&
                image_position.x < sensor.cols &&
                image_position.y < sensor.rows) {
              int idx = image_position.y * sensor.cols + image_position.x;
              if (mask[idx]) continue;
              if (depth[idx] < volume->GetOptions().min_sensor_depth) continue;
              if (depth[idx] > volume->GetOptions().max_sensor_depth) continue;
              float3 point3d = GetPoint3d(idx, depth[idx], sensor);
              float surface_distance = distance(start_pt, point3d);
              float voxel_distance = distance(start_pt, voxel_position);
              if (voxel_distance > surface_distance - truncation_distance &&
                  voxel_distance < surface_distance + truncation_distance &&
                  (depth[idx] < volume->GetOptions().max_sensor_depth)) {
                Voxel voxel;
                voxel.sdf = surface_distance - voxel_distance;
                voxel.color = color[idx];
                voxel.weight = (unsigned char)1;
                // To world coordinates
                voxel_position = transform * voxel_position;
                volume->UpdateVoxel(voxel_position, voxel);
              } else if (voxel_distance <
                         surface_distance - truncation_distance) {
                voxel_position = transform * voxel_position;
                Voxel voxel;
                voxel.sdf = truncation_distance;
                voxel.color = color[idx];
                voxel.weight = (unsigned char)1;
                volume->UpdateVoxel(voxel_position, voxel);
              }
            }
          }
        }
      }  // End single block update
    }
  }
}

void TsdfVolume::IntegrateScan(const RgbdImage &image, float4x4 camera_pose,
                               bool *mask) {
  int threads_per_block = THREADS_PER_BLOCK2;
  int thread_blocks =
      (options_.num_buckets * options_.bucket_size + threads_per_block - 1) /
      threads_per_block;

  tsdfvh::TsdfVolume* volume_d;
  cudaMalloc(&volume_d, sizeof(tsdfvh::TsdfVolume));
  cudaMemcpy(volume_d, this, sizeof(tsdfvh::TsdfVolume), cudaMemcpyHostToDevice);

  unsigned int hcv = 500000 - 1;
  unsigned int* heap_counter;
  cudaMalloc(&heap_counter, sizeof(unsigned int));
  cudaMemcpy(heap_counter, &hcv, sizeof(unsigned int), cudaMemcpyHostToDevice);

  AllocateFromDepthKernel<<<thread_blocks, threads_per_block>>>(
      volume_d, image.depth_, image.sensor_, camera_pose, heap_counter);
  cudaDeviceSynchronize();

  cudaFree(heap_counter);

  std::size_t mask_size = sizeof(bool) * image.sensor_.rows * image.sensor_.cols;

  bool* mask_d;
  cudaMalloc(&mask_d, mask_size);
  cudaMemcpy(mask_d, mask, mask_size, cudaMemcpyHostToDevice);

  float4x4 inv_camera_pose = camera_pose.getInverse();
  IntegrateScanKernel<<<thread_blocks, threads_per_block>>>(
      volume_d, image.rgb_, image.depth_, image.sensor_, camera_pose,
      inv_camera_pose, mask_d);
  cudaDeviceSynchronize();

  cudaMemcpy(this, volume_d, sizeof(tsdfvh::TsdfVolume), cudaMemcpyDeviceToHost);
  cudaFree(volume_d);

  cudaMemcpy(mask, mask_d, mask_size, cudaMemcpyDeviceToHost);
  cudaFree(mask_d);
}

__global__ void GenerateDepthKernel(TsdfVolume *volume, RgbdSensor sensor,
                                    float4x4 camera_pose,
                                    float *virtual_depth, unsigned int* calls) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size = sensor.rows * sensor.cols;

  float3 start_pt =
      make_float3(camera_pose.m14, camera_pose.m24, camera_pose.m34);
  for (int i = index; i < size; i += stride) {
    float current_depth = 0;
    while (current_depth < volume->GetOptions().max_sensor_depth) {
      float3 point = GetPoint3d(i, current_depth, sensor);
      point = camera_pose * point;
      Voxel v = volume->GetInterpolatedVoxel(point, i, calls);
      if (v.weight == 0) {
        current_depth += volume->GetOptions().truncation_distance;
      } else {
        current_depth += v.sdf;
      }
      if (v.weight != 0 && v.sdf < volume->GetOptions().voxel_size) break;
    }
    virtual_depth[i] = current_depth;
  }
}

__global__ void GenerateRgbKernel(TsdfVolume *volume, RgbdSensor sensor,
                                  float4x4 camera_pose, uchar3 *virtual_rgb,
                                  unsigned int* calls) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size = sensor.rows * sensor.cols;

  float3 start_pt =
      make_float3(camera_pose.m14, camera_pose.m24, camera_pose.m34);
  for (int i = index; i < size; i += stride) {
    float current_depth = 0;
    while (current_depth < volume->GetOptions().max_sensor_depth) {
      float3 point = GetPoint3d(i, current_depth, sensor);
      point = camera_pose * point;
      Voxel v = volume->GetInterpolatedVoxel(point, i, calls);
      if (v.weight == 0) {
        current_depth += volume->GetOptions().truncation_distance;
      } else {
        current_depth += v.sdf;
      }
      if (v.weight != 0 && v.sdf < volume->GetOptions().voxel_size) break;
    }
    if (current_depth < volume->GetOptions().max_sensor_depth) {
      float3 point = GetPoint3d(i, current_depth, sensor);
      point = camera_pose * point;
      Voxel v = volume->GetInterpolatedVoxel(point, i, calls);
      virtual_rgb[i] = v.color;
    } else {
      virtual_rgb[i] = make_uchar3(0, 0, 0);
    }
  }
}

float* TsdfVolume::GenerateDepth(float4x4 camera_pose, RgbdSensor sensor) {
  std::size_t depth_n = sensor.rows * sensor.cols;
  std::size_t depth_size = sizeof(float) * depth_n;
  
  float* virtual_depth_h = new float[depth_n];
  float* virtual_depth_d;
  cudaMalloc(&virtual_depth_d, depth_size);

  tsdfvh::TsdfVolume* volume_d;
  cudaMalloc(&volume_d, sizeof(tsdfvh::TsdfVolume));
  cudaMemcpy(volume_d, this, sizeof(tsdfvh::TsdfVolume), cudaMemcpyHostToDevice);

  int threads_per_block = THREADS_PER_BLOCK2;
  int thread_blocks =
      (sensor.rows * sensor.cols + threads_per_block - 1) / threads_per_block;

  const unsigned int total_threads = thread_blocks * threads_per_block;
  unsigned int* cache_calls_h = new unsigned int[total_threads];
  unsigned int* cache_calls_d;
  cudaMalloc(&cache_calls_d, total_threads * sizeof(unsigned int));
  cudaMemcpy(cache_calls_d, 0, total_threads * sizeof(unsigned int), cudaMemcpyHostToDevice);

  GenerateDepthKernel<<<thread_blocks, threads_per_block>>>(
      volume_d, sensor, camera_pose, virtual_depth_d, cache_calls_d);
  cudaDeviceSynchronize();

  cudaMemcpy(cache_calls_h, cache_calls_d, total_threads * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(cache_calls_d);

  cudaMemcpy(this, volume_d, sizeof(tsdfvh::TsdfVolume), cudaMemcpyDeviceToHost);
  cudaFree(volume_d);

  cudaMemcpy(virtual_depth_h, virtual_depth_d, depth_size, cudaMemcpyDeviceToHost);
  cudaFree(virtual_depth_d);

  return virtual_depth_h;
}

uchar3* TsdfVolume::GenerateRgb(float4x4 camera_pose, RgbdSensor sensor) {
  std::size_t rgb_n = sensor.rows * sensor.cols;
  std::size_t rgb_size = sizeof(uchar3) * rgb_n;

  uchar3* virtual_rgb_h = new uchar3[rgb_n];
  uchar3* virtual_rgb_d;
  cudaMalloc(&virtual_rgb_d, rgb_size);

  tsdfvh::TsdfVolume* volume_d;
  cudaMalloc(&volume_d, sizeof(tsdfvh::TsdfVolume));
  cudaMemcpy(volume_d, this, sizeof(tsdfvh::TsdfVolume), cudaMemcpyHostToDevice);

  int threads_per_block = THREADS_PER_BLOCK2;
  int thread_blocks =
      (sensor.rows * sensor.cols + threads_per_block - 1) / threads_per_block;

  const unsigned int total_threads = thread_blocks * threads_per_block;
  unsigned int* cache_calls_h = new unsigned int[total_threads];
  unsigned int* cache_calls_d;
  cudaMalloc(&cache_calls_d, total_threads * sizeof(unsigned int));
  cudaMemcpy(cache_calls_d, 0, total_threads * sizeof(unsigned int), cudaMemcpyHostToDevice);

  GenerateRgbKernel<<<thread_blocks, threads_per_block>>>(
      volume_d, sensor, camera_pose, virtual_rgb_d, cache_calls_d);
  cudaDeviceSynchronize();

  cudaMemcpy(cache_calls_h, cache_calls_d, total_threads * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(cache_calls_d);

  unsigned int total_calls = 0;
  for (unsigned int i = 0; i < total_threads; i++) {
    total_calls += cache_calls_h[i];
  }

  std::cout << "GenerateRgb calls: " << total_calls << std::endl;

  cudaMemcpy(this, volume_d, sizeof(tsdfvh::TsdfVolume), cudaMemcpyDeviceToHost);
  cudaFree(volume_d);

  cudaMemcpy(virtual_rgb_h, virtual_rgb_d, rgb_size, cudaMemcpyDeviceToHost);
  cudaFree(virtual_rgb_d);

  delete[] cache_calls_h;

  return virtual_rgb_h;
}

Mesh TsdfVolume::ExtractMesh(const float3 &lower_corner,
                             const float3 &upper_corner) {
  MeshExtractor *mesh_extractor = new MeshExtractor;
  mesh_extractor->Init(2000000, options_.voxel_size);
  mesh_extractor->ExtractMesh(this, lower_corner, upper_corner);
  Mesh *mesh;
  *mesh = mesh_extractor->GetMesh();
  delete mesh_extractor;
  return *mesh;
}

__host__ __device__ TsdfVolumeOptions TsdfVolume::GetOptions() {
  return options_;
}

}  // namespace tsdfvh

// RGBD_IMAGE.CU ----------------------------------------------------------------

RgbdImage::~RgbdImage() {
  cudaDeviceSynchronize();
  cudaFree(rgb_);
  cudaFree(depth_);
}

void RgbdImage::Init(const RgbdSensor &sensor) {
  sensor_ = sensor;
  cudaMalloc(&rgb_, sizeof(uchar3) * sensor_.rows * sensor.cols);
  cudaMalloc(&depth_, sizeof(float) * sensor_.rows * sensor.cols);
  cudaDeviceSynchronize();
}

__host__ __device__ inline float3 RgbdImage::GetPoint3d(int u, int v) const {
  float3 point;
  point.z = depth_[v * sensor_.cols + u];
  point.x = (static_cast<float>(u) - sensor_.cx) * point.z / sensor_.fx;
  point.y = (static_cast<float>(v) - sensor_.cy) * point.z / sensor_.fy;
  return point;
}

__host__ __device__ inline float3 RgbdImage::GetPoint3d(int i) const {
  int v = i / sensor_.cols;
  int u = i - sensor_.rows * v;
  return GetPoint3d(u, v);
}

// TRACKER.CU -------------------------------------------------------------------

Tracker::Tracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
                 const TrackerOptions &tracker_options,
                 const RgbdSensor &sensor) {
  volume_ = new tsdfvh::TsdfVolume;
  volume_->Init(tsdf_options);
  options_ = tracker_options;
  sensor_ = sensor;
  pose_ = Eigen::Matrix4d::Identity();
}

Tracker::~Tracker() {
  volume_->Free();
  delete volume_;
}

Eigen::Matrix4d v2t(const Vector6d &xi) {
  Eigen::Matrix4d M;

  M << 0.0  , -xi(2),  xi(1), xi(3),
       xi(2), 0.0   , -xi(0), xi(4),
      -xi(1), xi(0) , 0.0   , xi(5),
       0.0,   0.0   , 0.0   ,   0.0;

  return M;
}

__host__ __device__ float Intensity(float3 color) {
  return 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;
}

__host__ __device__ float ColorDifference(uchar3 c1, uchar3 c2) {
  float3 c1_float = ColorToFloat(c1);
  float3 c2_float = ColorToFloat(c2);
  return Intensity(c1_float)-Intensity(c2_float);
}

__global__ void CreateLinearSystem(tsdfvh::TsdfVolume *volume,
                                   float huber_constant, uchar3 *rgb,
                                   float *depth, bool *mask, float4x4 transform,
                                   RgbdSensor sensor, mat6x6 *acc_H,
                                   mat6x1 *acc_b, int downsample,
                                   float residuals_threshold,
                                   bool create_mask,
                                   unsigned int* calls) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size = sensor.rows * sensor.cols;
  for (int idx = index; idx < size / (downsample * downsample); idx += stride) {
    mat6x6 new_H;
    mat6x1 new_b;
    new_H.setZero();
    new_b.setZero();
    int v = (idx / (sensor.cols/downsample)) * downsample;
    int u = (idx - (sensor.cols/downsample) * v/downsample) * downsample;
    int i = v * sensor.cols + u;
    if (depth[i] < volume->GetOptions().min_sensor_depth) {
      continue;
    }
    if (depth[i] > volume->GetOptions().max_sensor_depth) {
      continue;
    }
    float3 point = transform * GetPoint3d(i, depth[i], sensor);
    tsdfvh::Voxel v1 = volume->GetInterpolatedVoxel(point, idx, calls);
    if (v1.weight == 0) {
      continue;
    }
    float sdf = v1.sdf;
    float3 color = make_float3(static_cast<float>(v1.color.x)/255,
                               static_cast<float>(v1.color.y)/255,
                               static_cast<float>(v1.color.z)/255);
    float3 color2 = make_float3(static_cast<float>(rgb[i].x)/255,
                                static_cast<float>(rgb[i].y)/255,
                                static_cast<float>(rgb[i].z)/255);
    if (sdf * sdf > residuals_threshold) {
      if (create_mask) mask[i] = true;
      continue;
    }
    mat1x3 gradient, gradient_color;
    // x
    float voxel_size = volume->GetOptions().voxel_size;
    v1 = volume->GetInterpolatedVoxel(point +
                                      make_float3(voxel_size, 0.0f, 0.0f), i, calls);
    if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    tsdfvh::Voxel v2 = volume->GetInterpolatedVoxel(
        point + make_float3(-voxel_size, 0.0f, 0.0f), i, calls);
    if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    gradient(0) = (v1.sdf - v2.sdf) / (2 * voxel_size);
    gradient_color(0) = ColorDifference(v1.color, v2.color) / (2 * voxel_size);
    // y
    v1 = volume->GetInterpolatedVoxel(point +
                                      make_float3(0.0f, voxel_size, 0.0f), i, calls);
    if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    v2 = volume->GetInterpolatedVoxel(point +
                                      make_float3(0.0f, -voxel_size, 0.0f), i, calls);
    if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    gradient(1) = (v1.sdf - v2.sdf) / (2 * voxel_size);
    gradient_color(1) = ColorDifference(v1.color, v2.color) / (2 * voxel_size);
    // z
    v1 = volume->GetInterpolatedVoxel(point +
                                      make_float3(0.0f, 0.0f, voxel_size), i, calls);
    if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    v2 = volume->GetInterpolatedVoxel(point +
                                      make_float3(0.0f, 0.0f, -voxel_size), i, calls);
    if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
      continue;
    }
    gradient(2) = (v1.sdf - v2.sdf) / (2 * voxel_size);
    gradient_color(2) = ColorDifference(v1.color, v2.color) / (2 * voxel_size);

    // Partial derivative of position wrt optimization parameters
    mat3x6 d_position;
    d_position(0, 0) = 0;
    d_position(0, 1) = point.z;
    d_position(0, 2) = -point.y;
    d_position(0, 3) = 1;
    d_position(0, 4) = 0;
    d_position(0, 5) = 0;
    d_position(1, 0) = -point.z;
    d_position(1, 1) = 0;
    d_position(1, 2) = point.x;
    d_position(1, 3) = 0;
    d_position(1, 4) = 1;
    d_position(1, 5) = 0;
    d_position(2, 0) = point.y;
    d_position(2, 1) = -point.x;
    d_position(2, 2) = 0;
    d_position(2, 3) = 0;
    d_position(2, 4) = 0;
    d_position(2, 5) = 1;

    // Jacobian
    mat1x6 jacobian = gradient * d_position;
    mat1x6 jacobian_color = gradient_color * d_position;

    float huber = fabs(sdf) < huber_constant ? 1.0 : huber_constant/fabs(sdf);
    bool use_depth = true;
    bool use_color = true;
    float weight = 0.025;
    if (use_depth) {
      new_H = new_H + huber * jacobian.getTranspose() * jacobian;
      new_b = new_b + huber * jacobian.getTranspose() * sdf;
    }

    if (use_color) {
      new_H = new_H + weight * jacobian_color.getTranspose() * jacobian_color;
      new_b = new_b +
              weight * jacobian_color.getTranspose() *
                  (Intensity(color) - Intensity(color2));
    }

    for (int j = 0; j < 36; j++) atomicAdd(&((*acc_H)(j)), new_H(j));
    for (int j = 0; j < 6; j++) atomicAdd(&((*acc_b)(j)), new_b(j));
  }
}

void Tracker::TrackCamera(const RgbdImage &image, bool *mask,
                          bool create_mask) {
  Vector6d increment, prev_increment;
  increment << 0, 0, 0, 0, 0, 0;
  prev_increment = increment;

  mat6x6 *acc_H = new mat6x6;
  mat6x1 *acc_b = new mat6x1;
  cudaDeviceSynchronize();
  for (int lvl = 0; lvl < 3; ++lvl) {
    for (int i = 0; i < options_.max_iterations_per_level[lvl]; ++i) {
      Eigen::Matrix4d cam_to_world = v2t(increment).exp() * pose_;
      Eigen::Matrix4f cam_to_worldf = cam_to_world.cast<float>();
      float4x4 transform_cuda = float4x4(cam_to_worldf.data()).getTranspose();

      acc_H->setZero();
      acc_b->setZero();
      int threads_per_block = THREADS_PER_BLOCK3;
      int thread_blocks =
          (sensor_.cols * sensor_.rows + threads_per_block - 1) /
          threads_per_block;
      bool create_mask_now =
          (lvl == 2) && (i == (options_.max_iterations_per_level[2] - 1)) &&
          create_mask;

      float residuals_threshold = 0;
      residuals_threshold = volume_->GetOptions().truncation_distance *
                            volume_->GetOptions().truncation_distance / 2;
      if (!create_mask) {
        residuals_threshold = volume_->GetOptions().truncation_distance *
                              volume_->GetOptions().truncation_distance;
      }

      tsdfvh::TsdfVolume* volume_d;
      cudaMalloc(&volume_d, sizeof(tsdfvh::TsdfVolume));
      cudaMemcpy(volume_d, volume_, sizeof(tsdfvh::TsdfVolume), cudaMemcpyHostToDevice);

      std::size_t mask_size = sizeof(bool) * image.sensor_.rows * image.sensor_.cols;

      bool* mask_d;
      cudaMalloc(&mask_d, mask_size);
      cudaMemcpy(mask_d, mask, mask_size, cudaMemcpyHostToDevice);

      mat6x6 *acc_H_d = new mat6x6;
      mat6x1 *acc_b_d = new mat6x1;
      cudaMalloc(&acc_H_d, sizeof(mat6x6));
      cudaMalloc(&acc_b_d, sizeof(mat6x1));
      cudaMemcpy(acc_H_d, acc_H, sizeof(mat6x6), cudaMemcpyHostToDevice);
      cudaMemcpy(acc_b_d, acc_b, sizeof(mat6x1), cudaMemcpyHostToDevice);

      const unsigned int total_threads = thread_blocks * threads_per_block;
      unsigned int* cache_calls_h = new unsigned int[total_threads];
      unsigned int* cache_calls_d;
      cudaMalloc(&cache_calls_d, total_threads * sizeof(unsigned int));
      cudaMemcpy(cache_calls_d, 0, total_threads * sizeof(unsigned int), cudaMemcpyHostToDevice);

      // Kernel to fill in parallel acc_H and acc_b
      CreateLinearSystem<<<thread_blocks, threads_per_block>>>(
          volume_d, options_.huber_constant, image.rgb_, image.depth_, mask_d,
          transform_cuda, sensor_, acc_H_d, acc_b_d, options_.downsample[lvl],
          residuals_threshold, create_mask_now, cache_calls_d);
      cudaDeviceSynchronize();
    
      cudaMemcpy(cache_calls_h, cache_calls_d, total_threads * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaFree(cache_calls_d);

      unsigned int total_calls = 0;
      for (unsigned int i = 0; i < total_threads; i++) {
        total_calls += cache_calls_h[i];
      }

      std::cout << "TrackCamera calls: " << total_calls << std::endl;

      cudaMemcpy(acc_H, acc_H_d, sizeof(mat6x6), cudaMemcpyDeviceToHost);
      cudaMemcpy(acc_b, acc_b_d, sizeof(mat6x1), cudaMemcpyDeviceToHost);
      cudaFree(acc_H_d);
      cudaFree(acc_b_d);

      cudaMemcpy(mask, mask_d, mask_size, cudaMemcpyDeviceToHost);
      cudaFree(mask_d);

      cudaMemcpy(volume_, volume_d, sizeof(tsdfvh::TsdfVolume), cudaMemcpyDeviceToHost);
      cudaFree(volume_d);
  
      delete[] cache_calls_h;

      Eigen::Matrix<double, 6, 6> H;
      Vector6d b;
      for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) {
          H(r, c) = static_cast<double>((*acc_H)(r, c));
        }
      }
      for (int k = 0; k < 6; k++) {
        b(k) = static_cast<double>((*acc_b)(k));
      }
      double scaling = 1 / H.maxCoeff();
      b *= scaling;
      H *= scaling;
      H = H + options_.regularization * Eigen::MatrixXd::Identity(6, 6) * i;
      increment = increment - H.ldlt().solve(b);
      Vector6d change = increment - prev_increment;
      if (change.norm() <= options_.min_increment) break;
      prev_increment = increment;
    }
  }
  if (std::isnan(increment.sum())) increment << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  cudaFree(acc_H);
  cudaFree(acc_b);

  pose_ = v2t(increment).exp() * pose_;
  prev_increment_ = increment;
}

void ApplyMaskFlood(const cv::Mat &depth, cv::Mat &mask, float threshold) {
  int erosion_size = 15;
  cv::Mat erosion_kernel = cv::getStructuringElement(
  cv::MORPH_ELLIPSE, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
  cv::Point(erosion_size, erosion_size));
  cv::Mat eroded_mask;
  cv::erode(mask, eroded_mask, erosion_kernel);
  std::vector<std::pair<int, int>> mask_vector;
  for (int i = 0; i < depth.rows; i++) {
    for (int j = 0; j < depth.cols; j++) {
      mask.at<uchar>(i, j) = 0;
      if (eroded_mask.at<uchar>(i, j) > 0) {
        mask_vector.push_back(std::make_pair(i, j));
      }
    }
  }

  while (!mask_vector.empty()) {
    int i = mask_vector.back().first;
    int j = mask_vector.back().second;
    mask_vector.pop_back();
    if (depth.at<float>(i, j) > 0 && mask.at<uchar>(i, j) == 0) {
      float old_depth = depth.at<float>(i, j);
      mask.at<uchar>(i, j) = 255;
      if (i - 1 >= 0) {  // up
        if (depth.at<float>(i - 1, j) > 0 && mask.at<uchar>(i-1, j) == 0 &&
            fabs(depth.at<float>(i - 1, j) - old_depth) <
                threshold * old_depth) {
          mask_vector.push_back(std::make_pair(i - 1, j));
        }
      }
      if (i + 1 < depth.rows) {  // down
        if (depth.at<float>(i + 1, j) > 0 && mask.at<uchar>(i+1, j) == 0 &&
            fabs(depth.at<float>(i + 1, j) - old_depth) <
                threshold * old_depth) {
          mask_vector.push_back(std::make_pair(i + 1, j));
        }
      }
      if (j - 1 >= 0) {  // left
        if (depth.at<float>(i, j - 1) > 0 && mask.at<uchar>(i, j-1) == 0 &&
            fabs(depth.at<float>(i, j - 1) - old_depth) <
                threshold * old_depth) {
          mask_vector.push_back(std::make_pair(i, j - 1));
        }
      }
      if (j + 1 < depth.cols) {  // right
        if (depth.at<float>(i, j + 1) > 0 && mask.at<uchar>(i, j+1) == 0 &&
            fabs(depth.at<float>(i, j + 1) - old_depth) <
                threshold * old_depth) {
          mask_vector.push_back(std::make_pair(i, j + 1));
        }
      }
    }
  }
}

void Tracker::AddScan(const cv::Mat &rgb, const cv::Mat &depth) {
  RgbdImage image;
  image.Init(sensor_);

  std::size_t image_n = image.sensor_.rows * image.sensor_.cols;

  uchar3* img_rgb = new uchar3[image_n];
  float* img_depth = new float[image_n];

  // Linear copy for now
  for (int i = 0; i < image.sensor_.rows; i++) {
    for (int j = 0; j < image.sensor_.cols; j++) {
      img_rgb[i * image.sensor_.cols + j] =
          make_uchar3(rgb.at<cv::Vec3b>(i, j)(2), rgb.at<cv::Vec3b>(i, j)(1),
                      rgb.at<cv::Vec3b>(i, j)(0));
      img_depth[i * image.sensor_.cols + j] = depth.at<float>(i, j);
    }
  }

  cudaMemcpy(image.rgb_, img_rgb, sizeof(uchar3) * image_n, cudaMemcpyHostToDevice);
  cudaMemcpy(image.depth_, img_depth, sizeof(float) * image_n, cudaMemcpyHostToDevice);

  delete[] img_rgb;
  delete[] img_depth;

  bool* mask = new bool[image_n];
  for (int i = 0; i < image.sensor_.rows * image.sensor_.cols; i++) {
    mask[i] = false;
  }

  if (!first_scan_) {
    Eigen::Matrix4d prev_pose = pose_;
    TrackCamera(image, mask, true);

    cv::Mat cvmask(image.sensor_.rows, image.sensor_.cols, CV_8UC1);
    for (int i = 0; i < image.sensor_.rows; i++) {
      for (int j = 0; j < image.sensor_.cols; j++) {
        if (mask[i * image.sensor_.cols + j]) {
          cvmask.at<uchar>(i, j) = 255;
        } else {
          cvmask.at<uchar>(i, j) = 0;
        }
      }
    }

    ApplyMaskFlood(depth,cvmask,0.007);

    int dilation_size = 10;
    cv::Mat dilation_kernel = cv::getStructuringElement(
    cv::MORPH_ELLIPSE, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
    cv::Point(dilation_size, dilation_size));
    cv::dilate(cvmask, cvmask, dilation_kernel);

    for (int i = 0; i < image.sensor_.rows; i++) {
      for (int j = 0; j < image.sensor_.cols; j++) {
        if (cvmask.at<uchar>(i, j) > 0) {
          mask[i * image.sensor_.cols + j] = true;
        } else {
          mask[i * image.sensor_.cols + j] = false;
        }
      }
    }

    pose_ = prev_pose;
    TrackCamera(image, mask, false);
  } else {
    first_scan_ = false;
  }
  Eigen::Matrix4f posef = pose_.cast<float>();
  float4x4 pose_cuda = float4x4(posef.data()).getTranspose();

  volume_->IntegrateScan(image, pose_cuda, mask);

  delete[] mask;
}

Eigen::Matrix4d Tracker::GetCurrentPose() {
  return pose_;
}

tsdfvh::Mesh Tracker::ExtractMesh(const float3 &lower_corner,
                                  const float3 &upper_corner) {
  return volume_->ExtractMesh(lower_corner, upper_corner);
}

__global__ void GetVoxelTesterKernel(tsdfvh::TsdfVolume *volume) {
  unsigned int i;
  tsdfvh::Voxel v = volume->GetVoxel(make_float3(0, 0, 0), 0, &i);
  v.sdf = 1.5;
  v.color = make_uchar3(255, 133, 68);
  v.weight = 39;
  volume->SetVoxel(make_float3(234, 62.8, 56.1), v);
  v = volume->GetVoxel(make_float3(234, 62.8, 56.1), 0, &i);
  printf("sdf: %f\n", v.sdf);
  // printf("v {\n  sdf: %f\n  colour: %d, %d, %d\n  weight: %d\n}\n", v.sdf, v.color.x, v.color.y, v.color.z, v.weight);
}

void Tracker::GetVoxelTester() {
  tsdfvh::TsdfVolume* volume_d;
  cudaMalloc(&volume_d, sizeof(tsdfvh::TsdfVolume));
  cudaMemcpy(volume_d, this, sizeof(tsdfvh::TsdfVolume), cudaMemcpyHostToDevice);

  GetVoxelTesterKernel<<<1, 1>>>(volume_d);
  cudaDeviceSynchronize();
  
  cudaMemcpy(this, volume_d, sizeof(tsdfvh::TsdfVolume), cudaMemcpyDeviceToHost);
  cudaFree(volume_d);
}

cv::Mat Tracker::GenerateRgb(int width, int height) {
  Eigen::Matrix4f posef = pose_.cast<float>();
  float4x4 pose_cuda = float4x4(posef.data()).getTranspose();
  RgbdSensor virtual_sensor;
  virtual_sensor.rows = height;
  virtual_sensor.cols = width;
  virtual_sensor.depth_factor = sensor_.depth_factor;
  float factor_x = static_cast<float>(virtual_sensor.cols) /
                   static_cast<float>(sensor_.cols);
  float factor_y = static_cast<float>(virtual_sensor.rows) /
                   static_cast<float>(sensor_.rows);
  virtual_sensor.fx = factor_x * sensor_.fx;
  virtual_sensor.fy = factor_y * sensor_.fy;
  virtual_sensor.cx = factor_x * sensor_.cx;
  virtual_sensor.cy = factor_y * sensor_.cy;
  uchar3 *virtual_rgb = volume_->GenerateRgb(pose_cuda, virtual_sensor);

  cv::Mat cv_virtual_rgb(virtual_sensor.rows, virtual_sensor.cols, CV_8UC3);
  for (int i = 0; i < virtual_sensor.rows; i++) {
    for (int j = 0; j < virtual_sensor.cols; j++) {
      cv_virtual_rgb.at<cv::Vec3b>(i, j)[2] =
          virtual_rgb[i * virtual_sensor.cols + j].x;
      cv_virtual_rgb.at<cv::Vec3b>(i, j)[1] =
          virtual_rgb[i * virtual_sensor.cols + j].y;
      cv_virtual_rgb.at<cv::Vec3b>(i, j)[0] =
          virtual_rgb[i * virtual_sensor.cols + j].z;
    }
  }

  return cv_virtual_rgb;
}
}  // namespace refusion
