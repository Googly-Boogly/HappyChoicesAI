## Docker Troubleshooting Guide
### Common Issues and Solutions
### 1. Docker Command Not Found

Issue:
When running a Docker command, you get an error message stating that Docker is not recognized as an internal or external command.

Solution:

    Ensure Docker is installed correctly by following the installation guide.
    Verify that Docker is running. On Windows and macOS, Docker Desktop should be running in the system tray.

### 2. Docker Daemon Not Running

Issue:
You receive an error message stating that the Docker daemon is not running.

Solution:

Start the Docker daemon. On Windows and macOS, open Docker Desktop. On Linux, run the following command:

```bash
sudo systemctl start docker
```

Verify the Docker daemon is running:

```bash
sudo systemctl status docker
```

### 3. Permission Denied

Issue:
You encounter a "permission denied" error when running Docker commands.

Solution:

Add your user to the Docker group to run Docker without sudo:

```bash
sudo usermod -aG docker $USER
```

Then, log out and log back in for the changes to take effect.

### 4. Docker Compose Command Not Found

Issue:
You get an error message stating that docker-compose is not recognized.

Solution:

Ensure Docker Compose is installed. It is included with Docker Desktop on Windows and macOS. On Linux, install Docker Compose separately by following the [installation instructions](https://docs.docker.com/compose/install/linux/).

### 5. Build Failures

Issue:
The Docker container build process fails.

Solution:

    Check the error messages for specific issues and ensure all dependencies are available.

    Clear Docker caches and try rebuilding:

```bash
docker-compose down
docker-compose up --build --force-recreate
```
### 6. Container Not Starting

Issue:
The Docker container fails to start or exits immediately.

Solution:

Check the container logs for error messages:

```bash
docker-compose logs
```

Ensure all required environment variables (e.g., OpenAI API key) are correctly set in the .env file.

### 7. Network Issues

Issue:
The container cannot access external networks or services.

Solution:

    Verify your network settings and ensure the Docker network is configured correctly.
    Restart Docker and your network services.

### 8. Out of Disk Space

Issue:
You receive an error indicating insufficient disk space.

Solution:

    Remove unused Docker images and containers to free up space:

```bash
docker system prune -a
```

If you encounter any other issues, please refer to the official [Docker documentation](https://docs.docker.com/) or seek help from the community.