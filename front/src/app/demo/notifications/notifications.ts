import { Component, OnInit } from '@angular/core';
import { Notification, NotificationService } from '../../services/notificationService';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-notifications',
  standalone: true,

  imports: [FormsModule, CommonModule],
  templateUrl: './notifications.html',
  styleUrl: './notifications.scss'
})
export class Notifications  implements OnInit {

  notifications: Notification[] = [];
  loading = true;

  constructor(private notificationService: NotificationService) {}

  ngOnInit(): void {
    this.notificationService.getAllNotifications().subscribe({
      next: (data) => {
        this.notifications = data;
        this.loading = false;
      },
      error: (err) => {
        console.error('Erreur chargement notifications:', err);
        this.loading = false;
      }
    });
  }

  selectedNotification: Notification | null = null;

openDetails(id: string) {
  this.notificationService.getNotificationById(id).subscribe({
    next: (notif) => this.selectedNotification = notif,
    error: (err) => console.error('Erreur chargement notification:', err),
  });
}

closeDetails() {
  this.selectedNotification = null;
}

}
