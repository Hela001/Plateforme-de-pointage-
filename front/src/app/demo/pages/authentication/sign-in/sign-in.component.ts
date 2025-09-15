import { Component } from '@angular/core';
import { Router, RouterModule } from '@angular/router';
import { SharedModule } from 'src/app/theme/shared/shared.module';
import { UserAuthService } from 'src/app/services/UserAuthService';


@Component({
  selector: 'app-sign-in',
  standalone: true,
  imports: [SharedModule, RouterModule],
  templateUrl: './sign-in.component.html',
  styleUrls: ['./sign-in.component.scss']
})
export class SignInComponent {
  email = '';
  password = '';

  constructor(

    private router: Router,
    private userAuth: UserAuthService 
  ) {}

onSubmit() {
  if (!this.email || !this.password) {
    alert('Veuillez entrer un email et un mot de passe.');
    return;
  }

  this.userAuth.signIn(this.email, this.password).subscribe({
    next: (user) => {
      if (user && user.username && user.role) {
      
        alert('Bienvenue ' + user.username);

        this.userAuth.setUser(user); 
    
        this.router.navigate(['/analytics']);
      } else {
        alert("Données utilisateur incomplètes.");
      }
    },
    error: (err) => {
      console.error('Erreur lors de la connexion', err);
      alert('Email ou mot de passe incorrect.');
    }
  });
}

}

