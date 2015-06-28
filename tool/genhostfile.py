from plot.cluster_pb2 import ClusterProto
from google.protobuf import text_format
import sys
import os
if len(sys.argv) != 2:
  print "must provide cluster.conf"

fd=open(sys.argv[1], 'r')
cluster = ClusterProto()
text_format.Merge(str(fd.read()), cluster)
nworker_procs=cluster.nworker_groups*cluster.nworkers_per_group/cluster.nworkers_per_procs;
nserver_procs=cluster.nserver_groups*cluster.nservers_per_group/cluster.nservers_per_procs;
nprocs=0
if(cluster.server_worker_separate):
    nprocs=nworker_procs+nserver_procs;
else:
    nprocs=max(nworker_procs, nserver_procs);

workspace=os.path.split(sys.argv[1])[0]
with open(os.path.join(workspace, 'hostfile'), 'w') as fdd:
  for i in range(nprocs):
    fdd.write('localhost\n')
  fdd.flush()
  fdd.close()
