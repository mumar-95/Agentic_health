-- Run this if you already created chat tables and only need document tables.
-- Or re-run the full supabase/schema.sql (also safe).

create extension if not exists "pgcrypto";

create table if not exists public.hospital_documents (
  id uuid primary key default gen_random_uuid(),
  file_name text not null unique,
  file_type text not null,
  file_size_bytes bigint not null default 0,
  file_mtime timestamptz,
  content_text text not null default '',
  section_count integer not null default 0,
  chunk_count integer not null default 0,
  content_sha256 text,
  folder_fingerprint text,
  indexed_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists hospital_documents_file_name_idx
  on public.hospital_documents (file_name);

create index if not exists hospital_documents_indexed_at_idx
  on public.hospital_documents (indexed_at desc);

create table if not exists public.hospital_document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.hospital_documents (id) on delete cascade,
  chunk_index integer not null,
  content text not null,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create index if not exists hospital_document_chunks_document_id_idx
  on public.hospital_document_chunks (document_id, chunk_index);

alter table public.hospital_documents enable row level security;
alter table public.hospital_document_chunks enable row level security;
