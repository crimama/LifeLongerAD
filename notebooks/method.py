import re
import datetime

def parse_log_file(file_path):
    """
    로그 파일을 파싱하여 current_mean과 magnitude 값을 리스트로 추출합니다.

    Args:
        file_path (str): 로그 파일 경로.

    Returns:
        tuple: (current_means, magnitudes) 리스트를 담은 튜플.
               파싱에 실패하면 ([], [])를 반환.
    """
    current_means = []
    magnitudes = []
    widths = []

    # 정규 표현식 패턴 (이전과 동일)
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - Step (\d+): current_mean=(\d+\.?\d*), width=(\d+\.?\d*), magnitude=(\d+\.?\d*)"

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()  # Remove leading/trailing whitespace
                match = re.match(pattern, line)

                if match:
                    _, _, current_mean_str, width, magnitude_str = match.groups()

                    try:
                        current_mean = float(current_mean_str)
                        magnitude = float(magnitude_str)
                        current_means.append(current_mean)
                        magnitudes.append(magnitude)
                        widths.append(float(width))
                    except ValueError:
                        print(f"Warning: Could not convert values to float in line: {line}")
                        # Optionally, you might want to continue parsing other lines
                        # or raise the exception depending on your needs.

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return [], []  # Return empty lists if file not found
    except Exception as e:
        print(f"An error occurred: {e}")
        return [],[]

    return current_means, widths, magnitudes
