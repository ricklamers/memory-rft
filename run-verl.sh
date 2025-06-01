#!/bin/bash

# Remove existing container if it exists
docker rm -f verl 2>/dev/null || true

# Create and run the container with a command that keeps it running
docker run -d --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post4 tail -f /dev/null

# Wait a moment for the container to be ready
sleep 2

# Check if container is running
if docker ps | grep -q verl; then
    echo "Container 'verl' is running successfully!"
    echo "To enter the container, run: docker exec -it verl bash"
    echo "Or run this script with 'exec' argument to enter automatically"
    
    # If 'exec' argument is provided, enter the container
    if [ "$1" = "exec" ]; then
        docker exec -it verl bash
    fi
else
    echo "Failed to start container"
    docker logs verl
fi
