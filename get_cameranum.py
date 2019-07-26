# coding : utf-8
import pyrealsense2 as rs
def main():
    ctx = rs.context()
    devices = ctx.query_devices()
    print(devices[0])
    print(devices[1])
    pass


if __name__ == '__main__':
    main()