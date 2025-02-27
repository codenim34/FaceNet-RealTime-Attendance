import os

def create_student_folders():
    # List of students with their IDs and names
    students = [
        ("210042106", "Adib_Sakhawat"),
        ("210042107", "Md_Hasibur_Rahman_Alif"),
        ("210042111", "Nabila_Islam"),
        ("210042112", "Namisa_Najah_Raisa"),
        ("210042114", "Nazifa_Tasneem"),
        ("210042115", "Nafisa_Binte_Ghulam_Kibria"),
        ("210042117", "Takia_Farhin"),
        ("210042122", "Tasnim_Ashraf"),
        ("210042125", "Ishmaam_Iftekhar_Khan"),
        ("210042131", "Nusrat_Siddique"),
        ("210042132", "Antara_Arifa_Mullick"),
        ("210042133", "Md_Sakib_Hossain"),
        ("210042135", "Ahmed_Sadman_Labib"),
        ("210042137", "Tahsin_Islam"),
        ("210042143", "Hamim_Saad_Al_Raji"),
        ("210042146", "Taki_Tajwaruzzaman_Khan"),
        ("210042148", "Minhajul_Abedin_Bhuiyan"),
        ("210042149", "Md_Istiaq_Prodhan"),
        ("210042150", "Shat_El_Shahriar_Khan"),
        ("210042151", "Kashshaf_Labib"),
        ("210042155", "Md_Abid_Shahriar"),
        ("210042156", "Navid_Kamal"),
        ("210042163", "Faiza_Maliat"),
        ("210042166", "Syed_Md_Shadman_Alam"),
        ("210042167", "Ahabab_Imtiaz_Risat"),
        ("210042170", "Hasibul_Islam_Nirjhar"),
        ("210042172", "Adid_Al_Mahamud_Shazid"),
        ("210042173", "Md_Mainul_Hasan"),
        ("210042174", "Hasin_Mahtab_Alvee"),
        ("210042177", "Mehedi_Al_Mahmud")
    ]

    # Create dataset directory if it doesn't exist
    os.makedirs("dataset", exist_ok=True)

    # Create folders for each student
    for student_id, name in students:
        # Create folder name in the format: student_id_name
        folder_name = f"{student_id}_{name}"
        folder_path = os.path.join("dataset", folder_name)
        
        # Create the folder
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_name}")

        # Create a simple README in each folder
        readme_path = os.path.join(folder_path, "README.txt")
        with open(readme_path, "w") as f:
            f.write(f"Please add 5-10 clear face images of {name.replace('_', ' ')}\n")
            f.write("\nSupported formats: .jpg, .jpeg, .png\n")
            f.write("\nNote: Image names don't matter, just make sure they're in the correct folder!")

if __name__ == "__main__":
    print("Creating student folders...")
    create_student_folders()
    print("\nDone! Please add 5-10 face images for each student in their respective folders.") 