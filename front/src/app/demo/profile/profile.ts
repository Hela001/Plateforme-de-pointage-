import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { UserService } from 'src/app/services/user';
import { UserAuthService } from 'src/app/services/UserAuthService';

@Component({
  selector: 'app-profile',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './profile.html',
  styleUrls: ['./profile.scss'],
})
export class Profile implements OnInit {
  user: any = {
    username: '',
    email: '',
    password: '',
    phone: '',
    address: '',
    photoUrl: '',
    role: '',
  };

  confirmPassword: string = '';
  isEditing = false;
  showPassword = false;
  originalUser: any;
  selectedPhoto: File | undefined = undefined;


  constructor(private userService: UserService, private authService: UserAuthService) {}

  ngOnInit(): void {
    const userId = this.authService.getId();
    if (!userId) {
      console.error('ID utilisateur manquant');
      return;
    }

    this.userService.getUserProfile(userId).subscribe({
      next: (data) => {
        this.user = data;
        this.originalUser = JSON.parse(JSON.stringify(data));
      },
      error: (err) => {
        console.error('Erreur chargement profil', err);
      },
    });
  }

toggleEdit() {
  if (this.isEditing) {
 
    this.user = JSON.parse(JSON.stringify(this.originalUser));
    this.confirmPassword = '';
    this.selectedPhoto = undefined;
    this.isEditing = false;
  } else {
   
    this.originalUser = JSON.parse(JSON.stringify(this.user));
    this.isEditing = true;
    this.confirmPassword = '';
  }
}


  passwordsMatch(): boolean {
    return this.user.password === this.confirmPassword;
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedPhoto = input.files[0];

      const reader = new FileReader();
      reader.onload = () => {
        this.user.photoUrl = reader.result as string;
      };
      reader.readAsDataURL(this.selectedPhoto);
    }
  }

  saveChanges() {
    if (!this.passwordsMatch()) {
      alert('Les mots de passe ne correspondent pas !');
      return;
    }

    const userId = this.authService.getId();
    if (!userId) {
      console.error('ID utilisateur manquant');
      return;
    }

    this.userService.updateUser(userId, this.user, this.selectedPhoto).subscribe({
      next: () => {
        alert('Profil mis à jour avec succès');
        this.isEditing = false;
        this.originalUser = JSON.parse(JSON.stringify(this.user));
        this.confirmPassword = '';
        this.selectedPhoto = undefined;
      },
      error: (err) => {
        alert('Erreur lors de la mise à jour');
        console.error(err);
      },
    });
  }
}
