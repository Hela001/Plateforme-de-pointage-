import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { SharedModule } from 'src/app/theme/shared/shared.module';
import { NgbDropdownModule } from '@ng-bootstrap/ng-bootstrap';
import { UserService } from 'src/app/services/user'; // Assure-toi que ce service contient bien getUserById
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-form-elements',
  standalone: true,
  imports: [SharedModule, NgbDropdownModule, CommonModule, FormsModule],
  templateUrl: './form-elements.component.html',
  styleUrls: ['./form-elements.component.scss']
})
export class FormElementsComponent implements OnInit {
  user: any = {
    username: '',
    email: '',
    password: '',
    role: 'ADMIN',
    photo: ''
  };
  selectedFile!: File;
  userId: string = '';


constructor(
  private route: ActivatedRoute,
  private userService: UserService,
  private router: Router 
) {}

  ngOnInit() {
    this.route.paramMap.subscribe(params => {
      this.userId = params.get('id') || '';
      if (this.userId) {
        this.userService.getUserById(this.userId).subscribe({
          next: data => this.user = data,
          error: err => console.error('Erreur chargement user', err)
        });
      }
    });
  }

  onFileChange(event: any) {
    this.selectedFile = event.target.files[0];
  }

onSubmit() {
  if (this.userId) {
    this.userService.updateUser(this.userId, this.user, this.selectedFile).subscribe({
      next: () => {
        alert('Utilisateur mis à jour !');
        this.router.navigate(['/tables']); // <-- redirection
      },
      error: err => alert('Erreur : ' + err.message)
    });
  } else {
    alert('Aucun utilisateur sélectionné');
  }
}

}
