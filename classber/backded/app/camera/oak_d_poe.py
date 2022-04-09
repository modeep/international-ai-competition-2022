import depthai as dai

def oak_read():
    pipeline = dai.Pipeline()

    camRgb = pipeline.createColorCamera()

    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    device_info = dai.DeviceInfo()
    device_info.state = dai.XLinkDeviceState.X_LINK_BOOTLOADER
    device_info.desc.protocol = dai.XLinkProtocol.X_LINK_TCP_IP
    device_info.desc.name = "169.254.1.222"

    return dai.Device(pipeline, device_info)
