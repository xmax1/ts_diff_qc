
# from pyfig import Pyfig

# c = Pyfig(notebook= False, sweep= None, c_update= None, resource= 'Kubernetes')

test_c = """
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: mypod
	image: ubuntu
	resources:
	  limits:
		memory: 100Mi
		cpu: 100m
	  requests:
		memory: 100Mi
		cpu: 100m
	command: ["sh", "-c", "echo 'Im a new pod' && sleep infinity"]
"""

test_gpu_c = """
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod-example
spec:
  containers:
	-	name: gpu-container
	image: gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest
	command: ["sleep", "infinity"]
	resources:
		limits:
			nvidia.com/gpu: 1
"""

n_gpu 			= 1
namespace 		= 'scb-usra'
pod_name 		= 'gpu-pod-xmkqavqs'
container_name 	= 'gpu-container'
mount_path 		= '/scb-usra'
kub_image 		= 'gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest'



    
test_gpu_c = {
	'apiVersion': 'v1',
	'kind': 'Pod',
	'metadata': {'name': pod_name, 'namespace': namespace},
	'spec': {
		'containers': [
			{
			'name': container_name,
			'image': kub_image,
			'resources': {'limits': {'nvidia.com/gpu': n_gpu} },

			'command': ["/bin/bash","-c"], 
            'args': ['pip install tqdm; sleep infinity'],
            # 'args': ['conda env update --file env.yml; sleep infinity'],
			
			'volumeMounts': [
				{'mountPath': mount_path, 'name': mount_path.split('/')[-1]},
			]
			}
		],

		'volumes': [
			{'name': 'scb-usra', 'persistentVolumeClaim': {'claimName': 'scb-usra'}},
		],
        'restartPolicy': 'Never',
	}, # spec
    
}

import yaml
test_gpu_c = yaml.dump(test_gpu_c)
print(test_gpu_c)

# c.resource.cluster_submit(kub_c= test_c)
# c.resource.cluster_submit(kub_c= test_gpu_c)
# c.resource.cluster_submit()
