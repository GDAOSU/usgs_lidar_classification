@echo off
echo Stopping USGS LiDAR Classification container...

REM Stop the container using the fixed name
docker stop lidar_container

echo Docker container stopped successfully.
pause