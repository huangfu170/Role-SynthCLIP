def add_prefix_to_keys(original_dict, prefix):
    # 使用字典推导式创建一个新字典
    new_dict = {f"{prefix}{key}": value for key, value in original_dict.items()}
    return new_dict