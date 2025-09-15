import { Component, output } from '@angular/core';
import { NavContentComponent } from './nav-content/nav-content.component';
import { NavigationItems, NavigationItem } from './navigation'; // Import types & items
import { UserAuthService } from 'src/app/services/UserAuthService';

@Component({
  selector: 'app-navigation',
  imports: [NavContentComponent],
  templateUrl: './navigation.component.html',
  styleUrls: ['./navigation.component.scss']
})
export class NavigationComponent {
  windowWidth: number;
  NavMobCollapse = output();

  // Correctly typed array
  filteredNavigation: NavigationItem[] = [];

  constructor(private authService: UserAuthService) {
    this.windowWidth = window.innerWidth;
    const userRole = this.authService.getRole();
    this.filteredNavigation = this.filterItems(NavigationItems, userRole);
  }

  private filterItems(items: NavigationItem[], role: string): NavigationItem[] {
    return items
      .filter(item => !item['roles'] || item['roles'].includes(role)) // If roles property is missing, allow
      .map(item => ({
        ...item,
        children: item.children ? this.filterItems(item.children, role) : []
      }));
  }

  navMobCollapse() {
    if (this.windowWidth < 992) {
      this.NavMobCollapse.emit();
    }
  }
}
