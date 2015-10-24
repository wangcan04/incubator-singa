/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/


#include <glog/logging.h>
#include <sys/stat.h>
#include "singa/io/lmdb_store.h"

namespace singa {
namespace io {

const size_t LMDB_MAP_SIZE = 1099511627776;  // 1 TB
bool LMDBStore::Open(const std::string& source, Mode mode) {
  MDB_CHECK(mdb_env_create(&mdb_env_));
  MDB_CHECK(mdb_env_set_mapsize(mdb_env_, LMDB_MAP_SIZE));
  if (mode == kCreate) {
    CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir " << source << "failed";
  }
  int flags = 0;
  if (mode == kRead) {
    flags = MDB_RDONLY | MDB_NOTLS;
  }
  int rc = mdb_env_open(mdb_env_, source.c_str(), flags, 0664);
#ifndef ALLOW_LMDB_NOLOCK
  MDB_CHECK(rc);
#else
  if (rc == EACCES) {
    LOG(WARNING) << "Permission denied. Trying with MDB_NOLOCK ...";
    // Close and re-open environment handle
    mdb_env_close(mdb_env_);
    MDB_CHECK(mdb_env_create(&mdb_env_));
    // Try again with MDB_NOLOCK
    flags |= MDB_NOLOCK;
    MDB_CHECK(mdb_env_open(mdb_env_, source.c_str(), flags, 0664));
  } else {
    MDB_CHECK(rc);
  }
#endif
  LOG(INFO) << "Opened lmdb " << source;

  if (mode == kRead) {
    MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_));
    MDB_CHECK(mdb_dbi_open(mdb_txn_, NULL, 0, &mdb_dbi_));
    MDB_CHECK(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_));
    Seek(MDB_FIRST);
  } else if (mode == kCreate) {
    MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn_));
    MDB_CHECK(mdb_dbi_open(mdb_txn_, NULL, 0, &mdb_dbi_));
  }
  mode_ = mode;
  return true;
}

bool LMDBStore::Read(std::string* key, std::string* value) {
  CHECK_EQ(mode_, kRead);
  if (!valid_)
    return valid_;
  *key = string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  *value = string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  Seek(MDB_NEXT);
  return true;
}

bool LMDBStore::Write(const std::string& key, const std::string& value) {
  CHECK_NE(mode_, kRead);
  MDB_val mdb_key, mdb_value;
  mdb_key.mv_data = const_cast<char*>(key.data());
  mdb_key.mv_size = key.size();
  mdb_value.mv_data = const_cast<char*>(value.data());
  mdb_value.mv_size = value.size();
  MDB_CHECK(mdb_put(mdb_txn_, mdb_dbi_, &mdb_key, &mdb_value, 0));
  return true;
}

}  // namespace io
}  // namespace singa
