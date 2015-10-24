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

#ifndef SINGA_IO_LMDB_STORE_H_
#define SINGA_IO_LMDB_STORE_H_

#include <string>
#include <lmdb.h>
#include <glog/logging.h>
#include "singa/io/store.h"


namespace singa {
namespace io {
inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}
/**
 */
class LMDBStore : public Store {
 public:
  ~LMDBStore() { Close();}
  bool Open(const std::string& source, Mode mode) override;
  void Close() override {
    if (mdb_cursor_ !=nullptr) {
      mdb_cursor_close(mdb_cursor_);
      mdb_txn_abort(mdb_txn_);
      mdb_cursor_ = nullptr;
    }

    if (mdb_env_ != nullptr) {
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
      mdb_env_ = nullptr;
    }
  }
  bool Read(std::string* key, std::string* value) override;
  void SeekToFirst() override {
    Seek(MDB_FIRST);
  }
  bool Write(const std::string& key, const std::string& value) override;
  void Flush() override {
    MDB_CHECK(mdb_txn_commit(mdb_txn_));
  }

 private:
  void Seek(MDB_cursor_op op) {
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      valid_ = false;
    } else {
      MDB_CHECK(mdb_status);
      valid_ = true;
    }
  }

 private:
  Mode mode_;

  MDB_env* mdb_env_ = nullptr;
  MDB_txn* mdb_txn_ = nullptr;
  MDB_cursor* mdb_cursor_ = nullptr;
  MDB_dbi mdb_dbi_;
  MDB_val mdb_key_, mdb_value_;
  bool valid_ = false;
};

}  // namespace io
}  // namespace singa

#endif  // SINGA_IO_LMDB_STORE_H_
