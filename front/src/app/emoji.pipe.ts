import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'emoji',
  standalone: true
})
export class EmojiPipe implements PipeTransform {
  transform(value: string): string {
    if (!value) return '';
    return value
      .replace(/:\)/g, '😊')
      .replace(/:\(/g, '☹️')
      .replace(/<3/g, '❤️')
      .replace(/:D/g, '😃');
  }
}
