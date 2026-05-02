import os


if __name__ == "__main__":
    try:
        input_path = input("filepath (.srt): ").strip().strip('"')
        if not os.path.isfile(input_path):
            print(f"Error: The file '{input_path}' was not found.", file=sys.stderr)
            sys.exit(1)

        directory, filename = os.path.split(input_path)
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{base_name}.txt")
        i=1
        while os.path.isfile(output_path):
            output_path = os.path.join(directory, f"{base_name} ({i}).txt")
            i+=1
        
        with open(input_path, encoding='utf-8') as file:
            data = file.read()
        segments = data.split(' --> ')
        
        lines = ['']
        
        for s in segments[1:]:
            line = s.split('\n')[1]
            if line != lines[-1].strip():
                lines.append(line)
        
        data = '\n'.join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            void = file.write(data)
        
        print("\n\nProcess terminated gracefully.")
    except KeyboardInterrupt:
        print("\n\nProcess cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during setup: {e}")
        raise e