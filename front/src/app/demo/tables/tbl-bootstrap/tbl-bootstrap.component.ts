import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { SharedModule } from 'src/app/theme/shared/shared.module';
import { UserService } from 'src/app/services/user';
import { UserAuthService } from 'src/app/services/UserAuthService';

@Component({
  selector: 'app-tbl-bootstrap',
  standalone: true,
  imports: [SharedModule,RouterLink],
  templateUrl: './tbl-bootstrap.component.html',
  styleUrls: ['./tbl-bootstrap.component.scss']
})
export class TblBootstrapComponent implements OnInit {
  users: any[] = [];
  userRole: string = '';
  userId: string = '';

  constructor(
    private userService: UserService,
    private authService: UserAuthService,
    private router: Router
  ) {}

  ngOnInit() {
    
    this.userRole = this.authService.getRole();
    this.userId = this.authService.getId();

    this.loadUsers();
  }

  loadUsers() {
    this.userService.getAllUsers().subscribe({
      next: (res) => {
        let filteredUsers = res;

        if (this.userRole === 'SITE_SUPERVISOR') {
          filteredUsers = res.filter(u =>
            u.role === 'EMPLOYEE' && u.supervisorId === this.userId
          );
        } else if (this.userRole === 'ADMIN' || this.userRole === 'HR') {
           filteredUsers = res.filter(u => u.role !== 'ADMIN' );
        } else {
          filteredUsers = []; 
        }

        this.users = filteredUsers.map(u => {
          if (u.id) {
            u.photoUrl = `http://localhost:8081/users/photo/${u.id}`;
          }
          return u;
        });
      },
      error: (err) => console.error('Erreur chargement utilisateurs', err)
    });
  }

  editUser(user: any) {
    this.router.navigate(['/forms', user.id]);
  }

  deleteUser(user: any) {
    if (confirm(`Are you sure you want to delete user ${user.username} ?`)) {
      this.userService.deleteUser(user.id).subscribe({
        next: () => {
          alert('User deleted');
          this.loadUsers();
        },
        error: err => alert('Error deleting user: ' + err.message)
      });
    }
  }
}
