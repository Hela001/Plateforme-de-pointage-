import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Pointage } from './pointage';

describe('Pointage', () => {
  let component: Pointage;
  let fixture: ComponentFixture<Pointage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Pointage]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Pointage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
