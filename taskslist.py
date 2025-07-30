class ToDoList:
    def __init__(self):
        self.tasks = []

    def show_tasks(self):
        if not self.tasks:
            print("No tasks available.")
        else:
            for i, task in enumerate(self.tasks, start=1):
                print(f"{i}. {task}")

    def add_task(self):
        task = input("Enter a task: ")
        self.tasks.append(task)
        print("Task added successfully.")

    def delete_task(self):
        if not self.tasks:
            print("No tasks available.")
        else:
            self.show_tasks()
            try:
                task_number = int(input("Enter the task number to delete: "))
                if task_number > 0 and task_number <= len(self.tasks):
                    del self.tasks[task_number - 1]
                    print("Task deleted successfully.")
                else:
                    print("Invalid task number.")
            except ValueError:
                print("Invalid input.")

def main():
    todo = ToDoList()
    while True:
        print("\nTo-Do List App")
        print("1. Show tasks")
        print("2. Add task")
        print("3. Delete task")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            todo.show_tasks()
        elif choice == "2":
            todo.add_task()
        elif choice == "3":
            todo.delete_task()
        elif choice == "4":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
