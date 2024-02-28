Build

`docker build -t my-autogen-1 .`

Run

`docker run --name maug-1 -it my-autogen-1`

Run indefinitely `docker run -d -t my-autogen-1`
Connect to the container `docker exec -it 6d14f66f9c78 /bin/bash`

copy files
`docker cp CONTAINER:<dir> <host_dir>`
ex. `docker cp 6d14f66f9c78:/app/temp_dir ~/temp_img`
