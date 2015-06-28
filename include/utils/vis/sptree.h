/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#ifndef SPTREE_H
#define SPTREE_H

using namespace std;


class Cell {

    unsigned int dimension;
    float* corner;
    float* width;


public:
    Cell(unsigned int inp_dimension);
    Cell(unsigned int inp_dimension, float* inp_corner, float* inp_width);
    ~Cell();

    float getCorner(unsigned int d);
    float getWidth(unsigned int d);
    void setCorner(unsigned int d, float val);
    void setWidth(unsigned int d, float val);
    bool containsPoint(float point[]);
};


class SPTree
{

    // Fixed constants
    static const unsigned int QT_NODE_CAPACITY = 1;

    // A buffer we use when doing force computations
    float* buff;

    // Properties of this node in the tree
    SPTree* parent;
    unsigned int dimension;
    bool is_leaf;
    unsigned int size;
    unsigned int cum_size;

    // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
    Cell* boundary;

    // Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
    float* data;
    float* center_of_mass;
    unsigned int index[QT_NODE_CAPACITY];

    // Children
    SPTree** children;
    unsigned int no_children;

public:
    SPTree(unsigned int D, float* inp_data, unsigned int N);
    SPTree(unsigned int D, float* inp_data, float* inp_corner, float* inp_width);
    SPTree(unsigned int D, float* inp_data, unsigned int N, float* inp_corner, float* inp_width);
    SPTree(SPTree* inp_parent, unsigned int D, float* inp_data, unsigned int N, float* inp_corner, float* inp_width);
    SPTree(SPTree* inp_parent, unsigned int D, float* inp_data, float* inp_corner, float* inp_width);
    ~SPTree();
    void setData(float* inp_data);
    SPTree* getParent();
    void construct(Cell boundary);
    bool insert(unsigned int new_index);
    void subdivide();
    bool isCorrect();
    void rebuildTree();
    void getAllIndices(unsigned int* indices);
    unsigned int getDepth();
    void computeNonEdgeForces(unsigned int point_index, float theta, float neg_f[], float* sum_Q);
    void computeEdgeForces(unsigned int* row_P, unsigned int* col_P, float* val_P, int N, float* pos_f);
    void print();

private:
    void init(SPTree* inp_parent, unsigned int D, float* inp_data, float* inp_corner, float* inp_width);
    void fill(unsigned int N);
    unsigned int getAllIndices(unsigned int* indices, unsigned int loc);
    bool isChild(unsigned int test_index, unsigned int start, unsigned int end);
};

#endif
