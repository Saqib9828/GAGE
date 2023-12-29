def extract_features_counts(code_list):
    feature_types = [
        "Numeric constants",
        "String constants",
        "Transfer instructions",
        "Call instructions",
        "Arithmetic instructions",
        "Compare instructions",
        "Mov instructions",
        "Termination instructions",
        "Data declaration instructions",
    ]

    feature_counts = {feature_type: 0 for feature_type in feature_types}

    for line in code_list:
        words = line.split()

        if len(words) >= 2:
            instruction = words[0].lower()
            operand = words[1].lower()

            if instruction in (
            "push", "pop", "call", "jmp", "je", "jz", "jne", "jnz", "ja", "jae", "jb", "jbe", "jl", "jle", "jg", "jge",
            "jo", "jno", "js", "jns"):
                feature_counts["Transfer instructions"] += 1
            elif instruction == "mov":
                feature_counts["Mov instructions"] += 1
            elif instruction in ("add", "sub", "mul", "div", "inc", "dec", "and", "or", "xor", "not"):
                feature_counts["Arithmetic instructions"] += 1
            elif instruction in ("cmp", "test"):
                feature_counts["Compare instructions"] += 1
            elif instruction == "ret":
                feature_counts["Termination instructions"] += 1
            elif instruction in ("db", "dd", "dq"):
                feature_counts["Data declaration instructions"] += 1

            # Check for numeric constants
            for word in words:
                if word.startswith("0x") or word.isdigit():
                    feature_counts["Numeric constants"] += 1

            # Check for string constants
            if operand.startswith('"') and operand.endswith('"'):
                feature_counts["String constants"] += 1

    # Calculate the total number of instructions
    total_instructions = sum(feature_counts.values())

    # Append the total instructions count to the dictionary
    feature_counts["Total instructions"] = total_instructions

    return [feature_counts[feature_type] for feature_type in feature_types]


# Example usage:
list_x = [
                "push ebp",
                "mov ebp esp",
                "push esi",
                "push edi",
                "push ebx",
                "and esp 0FFFFFFF8h",
                "sub esp 50h",
                "mov eax [ebp+arg_10]",
                "mov ecx [ebp+arg_C]",
                "mov edx [ebp+arg_8]",
                "mov esi [ebp+arg_4]",
                "mov edi [ebp+arg_0]",
                "xor ebx ebx",
                "mov [esp+5Ch+var_28] eax",
                "mov eax [esp+5Ch+var_20]",
                "mov [esp+5Ch+var_2C] eax",
                "and eax 2DDE0057h",
                "mov [esp+5Ch+var_20] eax",
                "mov [esp+5Ch+var_24] 4FF87922h",
                "mov eax [esp+5Ch+var_1C]",
                "mov [esp+5Ch+var_30] eax",
                "mov eax [esp+5Ch+var_18]",
                "mov [esp+5Ch+var_34] eax",
                "mov eax [esp+5Ch+var_30]",
                "and eax 33EE0A08h",
                "test ecx ecx",
                "setnz cl",
                "mov [esp+5Ch+var_1C] eax",
                "mov [esp+5Ch+var_18] 0",
                "mov eax [esp+5Ch+var_28]",
                "cmp eax 0",
                "setz ch",
                "or cl ch",
                "test cl 1",
                "mov [esp+5Ch+var_38] edi",
                "mov [esp+5Ch+var_3C] edx",
                "mov [esp+5Ch+var_40] esi",
                "mov [esp+5Ch+var_44] ebx",
                "jnz loc_4010DA"
            ]

feature_counts = extract_features_counts(list_x)
print(feature_counts)
