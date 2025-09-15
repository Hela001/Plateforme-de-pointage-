import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Pointage, PointageResponse, PointageService } from 'src/app/services/pointageService';
import { UserAuthService } from 'src/app/services/UserAuthService';
import { UserService } from 'src/app/services/user';
import { RouterModule } from '@angular/router';
@Component({
  selector: 'app-pointage',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './pointage.html',
  styleUrls: ['./pointage.scss']
})
export class PointageComponent implements OnInit {
  resultat: PointageResponse | null = null;
  loading = false;
  error = '';
  pointages: Pointage[] = [];
  vue: 'form' | 'liste' = 'form';
  alerteFraude: string | null = null;
  raisonsFraude: string[] = [];
  fraude: boolean = false;
  message: string = '';
  fraudeVideo: boolean = false;
  raisonsVideo: string[] = [];

  userRole: string = '';
  userId: string = '';
  employeesSupervised: string[] = [];

  selectedPointage: Pointage | null = null;

  constructor(
    private pointageService: PointageService,
    private authService: UserAuthService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.userRole = this.authService.getRole();
    this.userId = this.authService.getId();

    if (this.userRole === 'SITE_SUPERVISOR') {
      this.userService.getAllUsers().subscribe(users => {
        this.employeesSupervised = users
          .filter(u => u.role === 'EMPLOYEE' && u.supervisorId === this.userId)
          .map(u => u.id);
        this.loadPointages();
      });
    } else {
      this.loadPointages();
    }
  }

  private loadPointages() {
    this.pointageService.getAllPointages().subscribe(data => {
      this.pointages = this.filterPointages(data);
    });
  }

  private filterPointages(pointages: Pointage[]): Pointage[] {
    if (this.userRole === 'ADMIN' || this.userRole === 'HR') return pointages;
    if (this.userRole === 'SITE_SUPERVISOR')
      return pointages.filter(p => this.employeesSupervised.includes(p.user_id));
    if (this.userRole === 'EMPLOYEE')
      return pointages.filter(p => p.user_id === this.userId);
    return [];
  }

  onVideoSelected(event: any, username: string) {
    const file = event.target.files[0];
    if (!file) return;

    this.loading = true;
    this.pointageService.detectFraude(file, username).subscribe({
      next: (res: any) => {
        console.log("Réponse détecteur fraude:", res);
        this.loading = false;

        if (res.fraude) {
          this.fraudeVideo = true;
          this.raisonsVideo = res.raisons || [];
          this.fraude = true;
          this.message = "🚨 Fraude Pointage détectée !";
          this.raisonsFraude = this.raisonsVideo;
        } else {
          this.fraudeVideo = false;
          this.raisonsVideo = [];
          this.fraude = false;
          this.message = "✅ Vidéo valide.";
          this.raisonsFraude = [];
        }
      },
      error: (err: any) => {
        this.loading = false;
        this.fraude = false;
        this.message = "Erreur : " + (err.error?.error || "Analyse échouée.");
        this.raisonsFraude = [];
      }
    });
  }

  fairePointage() {
    this.error = '';
    this.resultat = null;

    if (!navigator.geolocation) {
      this.error = 'Géolocalisation non supportée par ce navigateur.';
      return;
    }

    this.loading = true;
    navigator.geolocation.getCurrentPosition(
      position => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;

        this.pointageService.fairePointage(lat, lon).subscribe({
          next: res => {
            this.resultat = res;

            // Ne pas toucher aux valeurs de fraude vidéo ici
            if (res.fraude && !this.fraudeVideo) {
              this.fraude = true;
              this.raisonsFraude = res.raisons || [''];
              this.message = '';
            } else if (!res.fraude) {
              this.fraude = false;
              this.message = "✅ Aucune fraude détectée.";
              this.raisonsFraude = [];
            }

            this.loading = false;
          },
          error: () => {
            this.error = 'Erreur lors du pointage.';
            this.loading = false;
          }
        });
      },
      err => {
        this.error = `Erreur géolocalisation: ${err.message}`;
        this.loading = false;
      }
    );
  }

  private parseDate(value: any): Date | null {
    if (!value) return null;
    let dateStr: any;
    if (typeof value === 'object') {
      if ('$date' in value) {
        dateStr = value.$date;
      } else if ('$numberLong' in value) {
        dateStr = parseInt(value.$numberLong, 10);
      } else {
        return null;
      }
    } else {
      dateStr = value;
    }
    const d = new Date(dateStr);
    return isNaN(d.getTime()) ? null : d;
  }

  openDetails(pointage: Pointage) {
    const clone: any = { ...pointage };
    clone.arrivees = Array.isArray(clone.arrivees) ? clone.arrivees : (clone.arrivees ? [clone.arrivees] : []);
    clone.departs = Array.isArray(clone.departs) ? clone.departs : (clone.departs ? [clone.departs] : []);

    clone.arrivees = clone.arrivees
      .filter((a: any) => a.heure !== undefined && a.heure !== null)
      .map((a: any) => ({ ...a, heure: this.parseDate(a.heure) }));

    clone.departs = clone.departs
      .filter((d: any) => d.heure !== undefined && d.heure !== null)
      .map((d: any) => ({ ...d, heure: this.parseDate(d.heure) }));

    this.selectedPointage = clone;
  }

  closeDetails() {
    this.selectedPointage = null;
  }

  get arrivees(): { heure: Date | null }[] {
    return this.selectedPointage?.arrivees ?? [];
  }

  get departs(): { heure: Date | null }[] {
    return this.selectedPointage?.departs ?? [];
  }
}
