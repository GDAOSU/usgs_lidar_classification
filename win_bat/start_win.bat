@echo off
echo Starting Docker container for USGS LiDAR Classification...

REM Run the Docker container with a fixed name
docker run --gpus "all" -p 7860:7860 -it -d --name lidar_container --rm gdaosu/usgs_lidar_classification:v1.0

REM Wait for a few seconds to allow the container to start
timeout /t 3 /nobreak

REM Open the default browser to localhost:7860
start http://localhost:7860

echo Docker container started as 'lidar_container'
echo Browser opened at http://localhost:7860
pause