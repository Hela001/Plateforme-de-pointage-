import { Component } from '@angular/core';
import { UserService } from '../../services/user';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

@Component({
  selector: 'app-forgot-password',
  standalone: true,
  imports: [FormsModule,CommonModule],
  templateUrl: './forgot-password.html',
  styleUrl: './forgot-password.scss'
})
export class ForgotPasswordComponent {
  email = '';
  code = '';
  newPassword = '';
  step = 1;
  message = '';
  passwordMismatch = false;

  confirmPassword = '';



constructor(private userService: UserService, private router: Router) {}

  sendCode() {
    this.userService.forgotPassword(this.email).subscribe({
      next: (res: any) => { this.message = res.message; this.step = 2; },
      error: (err) => { this.message = err.error.message; }
    });
  }

 resetPassword() {
  if (this.newPassword !== this.confirmPassword) {
    this.passwordMismatch = true;
    return;
  }

  this.passwordMismatch = false;

  this.userService.resetPassword(this.email, this.code, this.newPassword).subscribe({
    next: (res: any) => {
      this.message = res.message;
      this.step = 3;

      // 🔹 Redirection automatique après 3 secondes
      setTimeout(() => {
        this.router.navigate(['/auth/signin']);
      }, 3000);
    },
    error: (err) => {
      this.message = err.error.message;
    }
  });
}

}