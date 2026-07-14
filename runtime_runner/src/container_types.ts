export interface RegisteredWorkspace {
  name: string;
  folder: string;
  trigger: string;
  added_at: string;
  requiresTrigger?: boolean; // Default: true for groups, false for solo chats
}
