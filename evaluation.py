def calculate_translation_accuraccy(system_prediction, correct_answer):
    sys_pre_array = system_prediction.split()
    cor_ans_array = correct_answer.split()

    correct_count = 0
    for i in range(len(sys_pre_array)):
        if sys_pre_array[i] == cor_ans_array[i]:
            correct_count += 1

    accuracy = correct_count / len(sys_pre_array) * 100
    return accuracy


if __name__ == "__main__":
    # Example usage
    system_prediction = "the cat sat on the mat"
    correct_answer = "the cat sat on the mat"
    accuracy = calculate_translation_accuraccy(system_prediction, correct_answer)
    print(f"Accuracy: {accuracy:.2f}%")