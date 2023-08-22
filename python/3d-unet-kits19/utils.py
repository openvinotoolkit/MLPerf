#! /usr/bin/env python3
# coding=utf-8

def parse_devices(device_string):
    result = []
    target_device = device_string.partition(":")[0]
    result.append(target_device)
    if device_string.find(":") != -1:
        hw_devices_str = device_string.partition(":")[-1]
        for hw_device in hw_devices_str.split(','):
            if hw_device[0] == '-':
                hw_device = hw_device[1:]
            result.append(hw_device)
    return result


def parse_value_per_device(devices, values_string, value_type):
    # Format: <device1>:<value1>,<device2>:<value2> or just <value>
    result = {}
    if not values_string:
        return result
    device_value_strings = values_string.split(',')
    for device_value_string in device_value_strings:
        device_value_vec = device_value_string.split(':')
        if len(device_value_vec) == 2:
            device_name = device_value_vec[0]
            value = device_value_vec[1]
            if device_name in devices:
                result[device_name] = value
            else:
                devices_str = ""
                for device in devices:
                    devices_str += device + " "
                devices_str = devices_str.strip()
                raise Exception(f"Failed to set property to '{device_name}' "
                                f"which is not found in the target devices list '{devices_str}'!")
        elif len(device_value_vec) == 1:
            value = device_value_vec[0]
            for device in devices:
                result[device] = value
        elif not device_value_vec:
            raise Exception('Unknown string format: ' + values_string)
    return result
